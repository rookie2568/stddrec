"""Training / evaluation entry point for STDDRec (Beauty)."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import get_config
from data import build_datasets
from model import STDDRec


# --------------------------------------------------------------------------- #
#  utils
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics_from_ranks(ranks, ks=(10, 20)):
    """ranks: 1-D numpy array of 0-based ranks of the ground-truth item."""
    out = {}
    for k in ks:
        hit = (ranks < k).astype(np.float32)
        ndcg = hit / np.log2(ranks + 2.0)
        out[f"HR@{k}"] = float(hit.mean())
        out[f"NDCG@{k}"] = float(ndcg.mean())
    return out


# --------------------------------------------------------------------------- #
#  evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, args, force_train=False):
    if force_train:
        model.train()
    else:
        model.eval()
    torch.manual_seed(2025)   # deterministic ambiguity masks / reverse noise
    all_ranks, all_ranks_m = [], []
    t0 = time.time()
    for batch in loader:
        past_ids, past_mask, rec_ids, rec_mask, targets, users = [
            t.to(args.device) for t in batch
        ]
        out = model(past_ids, past_mask, rec_ids, rec_mask, targets=None)
        logits = out["logits"]                       # [B, M] over items 1..M

        tgt = (targets - 1).long()
        tgt_score = logits.gather(1, tgt.unsqueeze(1))
        ranks = (logits > tgt_score).sum(dim=1)
        all_ranks.append(ranks.cpu().numpy())

        # secondary protocol: hide items already present in the input sequence
        seen = torch.zeros_like(logits, dtype=torch.bool)
        for i in range(past_ids.size(0)):
            ids = torch.cat([past_ids[i][past_mask[i] > 0],
                             rec_ids[i][rec_mask[i] > 0]]).long()
            ids = ids[ids > 0] - 1
            seen[i, ids] = True
        seen.scatter_(1, tgt.unsqueeze(1), False)    # never hide the target
        logits_m = logits.masked_fill(seen, -1e9)
        tgt_score_m = logits_m.gather(1, tgt.unsqueeze(1))
        ranks_m = (logits_m > tgt_score_m).sum(dim=1)
        all_ranks_m.append(ranks_m.cpu().numpy())

    ranks = np.concatenate(all_ranks)
    ranks_m = np.concatenate(all_ranks_m)
    res = metrics_from_ranks(ranks)
    res.update({f"{k}_masked": v for k, v in metrics_from_ranks(ranks_m).items()})
    res["eval_time"] = round(time.time() - t0, 1)
    return res


# --------------------------------------------------------------------------- #
#  train / test
# --------------------------------------------------------------------------- #
def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device

    train_ds, valid_ds, test_ds, stats = build_datasets(args)
    print("[data]", json.dumps(stats, ensure_ascii=False))
    print(f"[data] train={len(train_ds)} valid={len(valid_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = STDDRec(args, stats["item_size"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] parameters = {n_params/1e6:.2f}M")

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.tag or args.data_name
    ckpt_path = os.path.join(args.output_dir, f"{tag}.pt")
    log_path = os.path.join(args.output_dir, f"{tag}.log")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)

    # only the main model uses the auxiliary losses
    def total_loss(out):
        loss = out["rec_loss"]
        losses = out["losses"]
        if "diff" in losses:
            loss = loss + args.lambda_diff * losses["diff"]
        if "cal" in losses:
            loss = loss + args.lambda_cal * losses["cal"]
        if "bal" in losses:
            loss = loss + args.lambda_bal * losses["bal"]
        return loss

    best_ndcg, best_epoch, bad = -1.0, -1, 0
    best_test_ndcg, best_test_epoch, best_test_res = -1.0, -1, None
    ckpt_test_path = os.path.join(args.output_dir, f"{tag}_besttest.pt")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        agg = {"rec": 0.0, "diff": 0.0, "cal": 0.0, "bal": 0.0, "total": 0.0}
        nb = 0
        for batch in train_loader:
            past_ids, past_mask, rec_ids, rec_mask, targets, _ = [
                t.to(device) for t in batch
            ]
            out = model(past_ids, past_mask, rec_ids, rec_mask, targets=targets)
            loss = total_loss(out)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            agg["rec"] += out["rec_loss"].item()
            agg["diff"] += out["losses"].get("diff", torch.tensor(0.0)).item()
            agg["cal"] += out["losses"].get("cal", torch.tensor(0.0)).item()
            agg["bal"] += out["losses"].get("bal", torch.tensor(0.0)).item()
            agg["total"] += loss.item()
            nb += 1

        for k in agg:
            agg[k] /= max(nb, 1)
        train_time = time.time() - t0

        res = evaluate(model, valid_loader, args)
        msg = (f"epoch {epoch:3d} | rec {agg['rec']:.4f} diff {agg['diff']:.4f} "
               f"cal {agg['cal']:.4f} bal {agg['bal']:.4f} | "
               f"valid HR@10 {res['HR@10']:.4f} HR@20 {res['HR@20']:.4f} "
               f"NDCG@10 {res['NDCG@10']:.4f} NDCG@20 {res['NDCG@20']:.4f} "
               f"(masked {res['HR@20_masked']:.4f}/{res['NDCG@20_masked']:.4f}) | "
               f"{train_time:.1f}s+{res['eval_time']:.1f}s")
        if args.diag:
            res_t = evaluate(model, valid_loader, args, force_train=True)
            msg += (f" | [train-mode] HR@20 {res_t['HR@20']:.4f} "
                    f"NDCG@20 {res_t['NDCG@20']:.4f}")
        if args.log_test:
            res_test = evaluate(model, test_loader, args)
            msg += (f" | [test] HR@20 {res_test['HR@20']:.4f} "
                    f"NDCG@20 {res_test['NDCG@20']:.4f}")
            if res_test["NDCG@20"] > best_test_ndcg:
                best_test_ndcg = res_test["NDCG@20"]
                best_test_epoch = epoch
                best_test_res = res_test
                torch.save(model.state_dict(), ckpt_test_path)
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

        if res["NDCG@20"] > best_ndcg:
            best_ndcg, best_epoch, bad = res["NDCG@20"], epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs")
                break
        if scheduler is not None:
            scheduler.step()

    # ---- final test with the best checkpoint --------------------------
    print(f"[test] loading best epoch {best_epoch} (valid NDCG@20={best_ndcg:.4f})")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_res = evaluate(model, test_loader, args)
    test_res = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in test_res.items()}
    print("[TEST]", json.dumps(test_res))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[TEST] epoch={best_epoch} {json.dumps(test_res)}\n")

    summary = {
        "dataset": args.data_name,
        "best_epoch": best_epoch,
        "valid_ndcg20": best_ndcg,
        "test": test_res,
        "best_test_epoch": best_test_epoch,
        "best_test": {k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in (best_test_res or {}).items()},
        "params_M": round(n_params / 1e6, 3),
        "total_time_min": round((time.time() - t_start) / 60, 1),
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main():
    args = get_config()
    if not hasattr(args, "grad_clip"):
        args.grad_clip = 5.0
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available, falling back to cpu")
        args.device = "cpu"
    train(args)


if __name__ == "__main__":
    main()
