from types import SimpleNamespace

import web.view_models as view_models


class FakeJob:
    def __init__(self, job_id: str, status: str) -> None:
        self.id = job_id
        self.status = status

    def public_data(self) -> dict:
        return {"id": self.id, "status": self.status}


def test_recent_jobs_keeps_active_jobs_outside_display_limit(monkeypatch) -> None:
    jobs = [FakeJob(str(index), "succeeded") for index in range(5)]
    jobs += [FakeJob("active", "running"), FakeJob("old", "failed")]
    service = SimpleNamespace(jobs=SimpleNamespace(recent=lambda: jobs))
    monkeypatch.setattr(view_models, "url_for", lambda endpoint, job_id: f"/jobs/{job_id}")

    visible = view_models._recent_jobs(service)

    assert [job["id"] for job in visible] == ["0", "1", "2", "3", "4", "active"]
