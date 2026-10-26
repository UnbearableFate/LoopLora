"""Main matrix contracts, blind release, and an actual tiny GPU collection."""
import copy
import json
from pathlib import Path

import pytest

from looplora.infra.seeds import stable_hash
from looplora.sym.main import make_plan, next_lower_lr, run_entry, save, seal


@pytest.mark.parametrize("steps", [15000, 20000])
def test_main_matrix_budgets_seeds_writers_and_evaluation_protocol(tmp_path, steps):
    plan = make_plan(tmp_path, steps, {"steps":steps})
    runs = plan["runs"]
    assert len(runs) == len({r["name"] for r in runs}) == 105
    assert {group:sum(r["group"]==group for r in runs) for group in
            ("writer","primary","replication","secondary")} == {
                "writer":3,"primary":72,"replication":18,"secondary":12}
    for row in runs:
        c=row["config"]
        assert c["train"]["steps"]==c["pretrain"]["steps"]==steps
        assert c["train"]["batch_size"]==64 and c["seed"]==row["seed"]
        if row["group"]=="writer":
            assert c["task"]["name"]=="t1" and c["task"]["head_sharing"]==.3
            assert c["train"]["lr"]==.003 and c["eval"]["splits"]==["val"]
        else:
            assert c["eval"]["depths"]==[0,1,2,3,4,6,8,12,16]
            assert c["eval"]["splits"]==["test","test_unseen"] and c["eval"]["seed"]==4242
            assert c["eval"]["swap_successors"]==(row["group"] in {"primary","replication"})
            assert c["train"]["depths"]==([1] if row["variant"]=="B4" else [1,2,4])
        if row["writer"]=="t1_frozen":
            assert len(row["dependencies"])==1
            assert row["dependencies"][0].endswith(f'-s{row["seed"]}')
            assert c["pretrain"]["writer_checkpoint"]==str(tmp_path/"runs"/row["dependencies"][0]/"writer.pt")
        if row["group"]=="secondary":
            assert row["seed"]==17 and row["r"]==16
    assert {r["config"]["train"]["lr"] for r in runs if r["writer"]=="joint" and r["variant"]=="M1-lin"}=={.001}


def test_nonfinite_retry_only_moves_one_grid_level():
    assert next_lower_lr(.003)==.001 and next_lower_lr(.001)==.0003
    with pytest.raises(ValueError,match="orchestrator"):
        next_lower_lr(.0003)


def make_receipts(tmp_path):
    plan=make_plan(tmp_path,15000,{"steps":15000})
    save(tmp_path/"plan.json",plan)
    for row in plan["runs"]:
        out=Path(row["run_dir"]); out.mkdir(parents=True)
        (out/"checkpoint.pt").touch()
        save(out/"record.json",dict(status="complete",training_complete=True,
            confirmation_samples_written=True,summary={"steps":15000},
            planned_config_hash=stable_hash(row["config"])))
        if row["group"]!="writer":
            (out/"confirmation").mkdir(); (out/"confirmation/samples.npz").touch()
            save(out/"confirmation/evaluation.json",{"aggregation_deferred":True})
    return plan


def test_barrier_requires_every_run_and_no_precomputed_confirmation_metrics(tmp_path):
    plan=make_receipts(tmp_path)
    first=Path(plan["runs"][3]["run_dir"])
    metric=first/"confirmation/metrics.json"; metric.write_text('{}')
    with pytest.raises(ValueError,match="before the global barrier"):
        seal(tmp_path/"plan.json")
    metric.unlink()
    record=json.loads((first/"record.json").read_text()); record["training_complete"]=False
    save(first/"record.json",record)
    with pytest.raises(ValueError,match="incomplete"):
        seal(tmp_path/"plan.json")
    record["training_complete"]=True; save(first/"record.json",record)
    result=seal(tmp_path/"plan.json")
    assert len(result["confirmation_samples"])==102 and len(result["runs"])==105
    assert result["plan_hash"]==stable_hash(plan)


def test_main_gpu_collection_defers_confirmation_and_preserves_failed_attempt(tmp_path, monkeypatch):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("GPU main-run canary requires compute")
    import looplora.sym.train as training
    entry=copy.deepcopy(make_plan(tmp_path,15000,{"steps":15000})["runs"][3])
    entry["config"]["train"].update(steps=2,batch_size=2,log_every=1)
    entry["config"]["eval"].update(episodes=4,batch_size=2)
    real_train=training.train
    calls=[]
    def first_nonfinite(config,out):
        calls.append(config["train"]["lr"])
        if len(calls)==1:
            (out/"train_log.jsonl").write_text('failed first minibatch\n')
            raise FloatingPointError("non-finite train loss at step 0")
        return real_train(config,out)
    monkeypatch.setattr(training,"train",first_nonfinite)
    record=run_entry(entry)
    folder=Path(entry["run_dir"])
    assert calls==[.003,.001] and record["status"]=="complete"
    assert (folder/"attempt-0/train_log.jsonl").read_text()=='failed first minibatch\n'
    assert (folder/"screening/metrics.json").exists()
    assert (folder/"confirmation/samples.npz").exists()
    assert not (folder/"confirmation/metrics.json").exists()
    metadata=json.loads((folder/"confirmation/evaluation.json").read_text())
    assert metadata["aggregation_deferred"] and "table" not in metadata and "diagnostics" not in metadata
    assert metadata["evaluation_seed"]==777001 and metadata["config"]["eval"]["splits"]==["test"]
    assert record["config"]["train"]["lr"]==.001
