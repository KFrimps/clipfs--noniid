def make_metric_logger(out_csv):
    # Turn string into a Path object
    out_csv = Path(out_csv)

    # Make sure the parent directory exists (runs/IID)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # If file doesn't exist yet, create it with headers
    if not out_csv.exists():
        pd.DataFrame(columns=["round", "accuracy", "loss"]).to_csv(out_csv, index=False)

    def aggregate_and_log(metrics):
        # metrics = list of (num_examples, {"accuracy": acc, "loss": loss})
        acc_sum, loss_sum, n_sum = 0.0, 0.0, 0
        for num_examples, m in metrics:
            acc_sum  += num_examples * m["accuracy"]
            loss_sum += num_examples * m["loss"]
            n_sum    += num_examples

        acc  = acc_sum / n_sum
        loss = loss_sum / n_sum

        # current round = number of existing rows + 1
        current_round = len(pd.read_csv(out_csv)) + 1

        # append new row
        pd.DataFrame([{
            "round": current_round,
            "accuracy": acc,
            "loss": loss
        }]).to_csv(out_csv, mode="a", header=False, index=False)

        print(f"[Round {current_round:02d}] GLOBAL acc={acc:.4f}, loss={loss:.4f}")
        return {"accuracy": acc, "loss": loss}

    return aggregate_and_log


class LogGlobalEvalFedAvg(FedAvg):
    """
    Logs per-client evaluation of the *global model* (after aggregation)
    to CSV each round. Works with clients that return:
        return loss, num_examples, {"accuracy": acc, "loss": loss}
    """
    def __init__(self, out_csv, **kwargs):
        super().__init__(**kwargs)
        self.out_csv = Path(out_csv)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)

        if not self.out_csv.exists():
            pd.DataFrame(
                columns=["round", "cid", "client_idx", "num_examples", "loss", "accuracy"]
            ).to_csv(self.out_csv, index=False)


    def aggregate_evaluate(self, rnd, results, failures):
        """results: List[Tuple[ClientProxy, EvaluateRes]]"""
        rows = []
        for client_proxy, ev_res in results:
            # Flower typically provides: ev_res.num_examples, ev_res.loss, ev_res.metrics
            m = ev_res.metrics or {}
            # Prefer metric dict values if present; fall back to ev_res.loss
            loss = float(m.get("loss", getattr(ev_res, "loss", float("nan"))))
            acc  = float(m.get("accuracy", float("nan")))
            
            client_idx = m.get("client_idx", None)

            rows.append([
                rnd,
                client_proxy.cid,
                client_idx,
                int(getattr(ev_res, "num_examples", 0)),
                loss,
                acc,
            ])


        if rows:
            pd.DataFrame(
                rows,
                columns=["round", "cid", "client_idx", "num_examples", "loss", "accuracy"]
            ).to_csv(self.out_csv, mode="a", header=False, index=False)

        # proceed with FedAvg's normal aggregation of eval metrics (e.g., weighted average)
        return super().aggregate_evaluate(rnd, results, failures)
