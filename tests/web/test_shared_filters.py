from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess

from pypdf import PdfReader
import pytest

from experiments.scenarios.adversarial import RANDOM_WALK_ENVIRONMENT, run_adversarial_experiment
from experiments.scenarios.adversarial_scaling import AdversarialScalingSpec, run_adversarial_scaling_experiment
from tests.web.support import create_test_app, csrf_token
from tests.web.test_figure_builder import context_for, digest_files, result
from web.validation import parse_profile_selection


@pytest.mark.parametrize("mode", ["fixed", "adversarial"])
def test_collection_reads_once_exports_selected_means_and_reuses_cache(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    app, service = create_test_app(Path("results"))
    if mode == "fixed":
        import experiments.plots.plot_regret as plotting
        profiles = ["hedge_vs_hedge", "ito_vs_ito"]
        for profile in profiles + ["bm_vs_bm"]:
            result(service, profile)
        loader_name = "load_rows"
    else:
        import experiments.plots.plot_adversarial as plotting
        profiles = ["auer_exp3", "ito"]
        for name in profiles + ["bm"]:
            for replicate in (0, 1):
                run_adversarial_experiment(name, n_actions=3, horizon=30, seed=42,
                    feedback_mode="bandit", environment=RANDOM_WALK_ENVIRONMENT,
                    replicate=replicate, output_dir=service.adversarial_raw_dir)
        loader_name = "load_adversarial_rows"
    originals = digest_files(tmp_path)
    loader = getattr(plotting, loader_name)
    reads, labels = Counter(), []
    original_save = plotting.save_figure_pair

    def load(path):
        reads[str(path)] += 1
        return loader(path)

    def save(figure, path, **kwargs):
        labels.append([line.get_label() for line in figure.axes[0].lines if not line.get_label().startswith("_")])
        return original_save(figure, path, **kwargs)

    monkeypatch.setattr(plotting, loader_name, load)
    monkeypatch.setattr(plotting, "save_figure_pair", save)
    monkeypatch.setattr(service, "submit_experiment", lambda *a, **k: pytest.fail("reran experiment"))
    context = context_for(service, mode, player=0)
    client = app.test_client()
    data = {"mode": mode, "context_id": context["id"], "profiles": list(reversed(profiles)),
            "_csrf_token": csrf_token(client)}
    response = client.post("/figure-builder/collection", data=data)
    assert response.status_code == 200
    figures = response.json["figures"]
    assert [(figure["metric"], figure["view"]) for figure in figures] == [
        (metric, view) for metric in ("external", "internal", "swap") for view in ("average", "sqrt_scaling")]
    assert len(reads) == 4 and set(reads.values()) == {1}
    assert labels == [["Hedge", "Ito"] if mode == "fixed" else ["AuerExp3", "Ito"]] * 6
    for figure in figures:
        assert figure["profiles"] == profiles
        preview, pdf = client.get(figure["url"]), client.get(figure["pdf_url"])
        assert preview.status_code == pdf.status_code == 200
        assert preview.data.startswith(b"\x89PNG")
        assert len(PdfReader(BytesIO(pdf.data)).pages) == 1
    timestamps = {p: p.stat().st_mtime_ns for p in service.figure_builder.output_dir.iterdir()}
    cached = client.post("/figure-builder/collection", data=data | {"profiles": profiles})
    assert cached.json == response.json
    assert set(reads.values()) == {1} and len(labels) == 6
    assert {p: p.stat().st_mtime_ns for p in timestamps} == timestamps
    chosen = [figures[5], figures[1], figures[3]]
    download = client.post("/figures/download-filtered.pdf", data={
        "mode": "figure_builder", "_csrf_token": data["_csrf_token"],
        "filenames": [figure["pdf_filename"] for figure in chosen],
    })
    assert download.status_code == 200 and download.mimetype == "application/pdf"
    pages = PdfReader(BytesIO(download.data)).pages
    assert len(pages) == 3
    for page, figure in zip(pages, chosen):
        original = PdfReader(service.figure_builder.artifact_path(figure["pdf_filename"])).pages[0]
        assert page.extract_text() == original.extract_text()
        assert not list(page.images)  # Merging keeps the publication PDFs vector-based.
    assert client.post("/figure-builder/collection", data=data | {"profiles": []}).status_code == 400
    assert client.post("/figure-builder/collection", data=data | {"profiles": ["unknown"]}).status_code == 400
    assert client.post("/figure-builder/collection", data=data | {"context_id": "a" * 24}).status_code == 400
    assert client.post("/figure-builder/collection", data={"mode": mode}).status_code == 400
    assert client.post("/figures/download-filtered.pdf", data={
        "mode": "figure_builder", "_csrf_token": data["_csrf_token"], "filenames": ["../private.pdf"],
    }).status_code == 404
    assert digest_files(tmp_path) == originals and not service.jobs.recent()


def run_ui(app, service, script, mode="fixed"):
    node = shutil.which("node")
    if not node or subprocess.run([node, "-e", "require('jsdom')"], capture_output=True).returncode:
        pytest.skip("Node.js with jsdom is unavailable")
    static = Path(__file__).parents[2] / "web/static"
    payload = {"page": app.test_client().get("/", query_string={"mode": mode}).get_data(as_text=True),
               "catalog": service.figure_builder.catalog(mode),
               "script": "\n".join((static / name).read_text() for name in ("common.js", "dashboard.js", "figure_builder.js"))}
    completed = subprocess.run([node, "-e", script], input=json.dumps(payload), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_shared_filters_six_figures_summary_export_and_stale_responses(tmp_path):
    app, service = create_test_app(tmp_path)
    for profile in ("hedge_vs_hedge", "ito_vs_ito", "hedge_vs_ito"):
        result(service, profile)
    result(service, "auer_exp3_vs_bm", mode="bandit")
    result(service, "bm_vs_bm", game="rpsls")
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window, d = w.document;
const f = name => d.getElementById("filter-" + name), b = name => d.getElementById("builder-" + name);
const tick = () => new Promise(resolve => setImmediate(resolve));
const change = (name, value) => {f(name).value = value; f(name).dispatchEvent(new w.Event("change"));};
const submit = () => d.getElementById("figure-builder").dispatchEvent(new w.Event("submit", {cancelable: true}));
const visibleRows = () => [...d.querySelectorAll(".summary-row")].filter(row => !row.hidden);
const visibleCards = () => [...b("figure").children].filter(card => !card.hidden);
const allFigures = ["external", "internal", "swap"].flatMap(metric => ["average", "sqrt_scaling"].map(view => ({
    metric, view, title: metric + " " + view, filename: metric + "_" + view + ".png",
    pdf_filename: metric + "_" + view + ".pdf", url: "/" + metric + "_" + view + ".png",
    pdf_url: "/" + metric + "_" + view + ".pdf"
})));
let requests = [], exports = [], finish, finishExport, downloads = 0;
w.HTMLElement.prototype.scrollIntoView = () => {};
w.HTMLAnchorElement.prototype.click = () => downloads++;
w.URL.createObjectURL = () => "blob:test";
w.URL.revokeObjectURL = () => {};
w.fetch = async (url, options) => {
    if (!options) return {ok: true, json: async () => payload.catalog};
    if (options.body.get("mode") === "figure_builder") {
        exports.push(options.body.getAll("filenames"));
        return new Promise(resolve => finishExport = () => resolve({ok: true, blob: async () => new w.Blob(["pdf"])}));
    }
    requests.push(options.body);
    return new Promise(resolve => finish = () => resolve({ok: true, json: async () => ({
        profiles: options.body.getAll("profiles"), figures: allFigures
    })}));
};
w.eval(payload.script);
(async () => {
    await tick(); await tick();
    assert.equal(b("generate").disabled, true);
    assert.equal(visibleRows().length, 0);
    assert.equal(d.querySelectorAll("#figure-builder select, .summary-panel select, .summary-panel input").length, 0);
    change("feedback", "full_information");
    f("select-all").click();
    assert.equal(visibleRows().length, 3);
    assert(visibleRows().every(row => row.dataset.player === "0"));
    submit();
    assert.equal(requests.length, 1);
    assert.deepEqual(requests[0].getAll("profiles"), ["hedge_vs_hedge", "hedge_vs_ito", "ito_vs_ito"]);
    assert.equal(requests[0].has("metric"), false);
    change("metric", "internal"); change("view", "sqrt_scaling");
    finish(); await tick(); await tick();
    assert.equal(b("figure").children.length, 6);
    assert.deepEqual(visibleCards().map(card => card.dataset.filename), ["internal_sqrt_scaling.png"]);
    assert.equal(requests.length, 1); // Display filters never regenerate the collection.
    for (const row of visibleRows()) {
        const visible = [...row.querySelectorAll("[data-regret]")].filter(cell => !cell.hidden);
        assert.equal(visible.length, 1);
        assert.equal(visible[0].dataset.metric, "sqrt_scaling_internal");
        const average = row.querySelector('[data-metric="average_internal"]');
        assert(Math.abs(Number(visible[0].dataset.value) - Number(average.dataset.value) * Math.sqrt(Number(row.dataset.horizon))) < 1e-12);
    }
    change("metric", "all"); change("view", "average");
    b("download").click();
    assert.deepEqual(exports[0], ["external_average.pdf", "internal_average.pdf", "swap_average.pdf"]);
    finishExport(); await tick(); await tick(); assert.equal(downloads, 1);
    b("download").click(); change("metric", "swap"); finishExport(); await tick(); await tick();
    assert.equal(downloads, 1); // Do not download an obsolete selection after filters change.
    f("clear-all").click();
    assert.equal(visibleRows().length, 0); assert.equal(b("figure").children.length, 0);
    assert.equal(b("download").disabled, true); submit(); assert.equal(requests.length, 1);
    f("profiles").value = "hedge_vs_ito";
    f("profiles").dispatchEvent(new w.Event("change"));
    assert.deepEqual(visibleRows().map(row => row.dataset.profile), ["hedge_vs_ito"]);
    submit(); f("clear-all").click(); finish(); await tick(); await tick();
    assert.equal(b("figure").children.length, 0);
    change("view", "all"); assert.equal(f("profiles").selectedOptions.length, 0);
    change("feedback", "bandit"); f("select-all").click(); change("player", "1"); f("select-all").click();
    assert.equal(visibleRows().length, 1);
    assert.equal(visibleRows()[0].dataset.profile, "auer_exp3_vs_bm");
    assert.equal(visibleRows()[0].dataset.player, "1");
    change("scope", "rpsls"); f("select-all").click();
    assert.equal(visibleRows().length, 1); assert.equal(visibleRows()[0].dataset.scope, "rpsls");
    f("clear-all").click();
    const persisted = JSON.parse(w.localStorage.getItem("swap-regret-shared-filters-fixed"));
    assert.deepEqual(persisted.selections[f("context").value], []);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script)


def test_scaling_only_results_remain_filterable(tmp_path):
    app, service = create_test_app(tmp_path)
    spec = AdversarialScalingSpec(environment=RANDOM_WALK_ENVIRONMENT, feedback_mode="bandit",
        algorithm_name="auer_exp3", action_counts=(3, 6), replicates=2, horizon=10,
        environment_seed=7, learner_seed=42)
    run_adversarial_scaling_experiment(spec, service.adversarial_scaling_raw_dir, workers=1)
    service.adversarial_scaling_figure_dir.mkdir(parents=True)
    (service.adversarial_scaling_figure_dir / f"{spec.run_id}_regret_by_actions.png").write_bytes(b"preview")
    script = r'''
const assert = require("assert").strict, {JSDOM} = require("jsdom");
const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
const dom = new JSDOM(payload.page, {url: "http://localhost/", runScripts: "outside-only"});
const w = dom.window, d = w.document;
w.fetch = async () => ({ok: true, json: async () => payload.catalog});
w.eval(payload.script);
(async () => {
    await new Promise(resolve => setImmediate(resolve));
    d.getElementById("filter-select-all").click();
    assert.equal(d.querySelector("[data-result-card]").hidden, false);
    assert.equal(d.getElementById("builder-generate").disabled, true);
    const metric = d.getElementById("filter-metric");
    metric.value = "swap"; metric.dispatchEvent(new w.Event("change"));
    assert.equal(d.querySelector("[data-result-card]").hidden, true);
    metric.value = "external"; metric.dispatchEvent(new w.Event("change"));
    assert.equal(d.querySelector("[data-result-card]").hidden, false);
    d.getElementById("filter-clear-all").click();
    assert.equal(d.querySelector("[data-result-card]").hidden, true);
    dom.window.close();
})().catch(error => {console.error(error); process.exit(1);});
'''
    run_ui(app, service, script, "adversarial")


def test_profile_selection_does_not_need_regret_or_view():
    selection = parse_profile_selection({"mode": "fixed", "context_id": "a" * 24,
                                         "profiles": ["ito_vs_ito", "hedge_vs_hedge", "ito_vs_ito"]})
    assert selection.profiles == ("hedge_vs_hedge", "ito_vs_ito")
