"""Smoke test: every page of the Streamlit app renders without exception (headless)."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_hash

from src import data_prep
from src.app import PAGES
from tests import synthetic

APP = str(Path(__file__).resolve().parents[1] / "src" / "app.py")
PATHS = [path for _, path, _ in PAGES]          # url path of each page, in the order of the navigation


def open_page(at: AppTest, path: str) -> AppTest:
    """Go to a page of st.navigation and run it. AppTest.switch_page only takes page files, and the app's pages
    are functions: st.Page identifies them by the hash of their url path."""
    at._page_hash = calc_hash(path)
    return at.run()


def _render_all_pages() -> None:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    titles = set()
    for path in PATHS:
        open_page(at, path)
        assert not at.exception, f"{path}: {[e.value for e in at.exception]}"
        titles.add(at.title[0].value)
    assert len(titles) == len(PATHS)                 # each page really ran, not the default one again


@pytest.fixture
def fresh_caches():
    """The app caches its dataset: start from, and leave, empty caches."""
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.mark.skipif(data_prep.find_dataset_dir() is None, reason="no dataset on disk (data/processed/real_v1)")
def test_all_pages_render(fresh_caches):
    """With the dataset the app loads (the real one, versioned in data/processed/real_v1)."""
    _render_all_pages()


def test_all_pages_render_on_synthetic_data(fresh_caches, monkeypatch):
    """With the synthetic test dataset (and its invented zoom), whatever is on disk: no data/ needed."""
    small = synthetic.build_dataset(with_zoom=True)
    monkeypatch.setattr(data_prep, "load_default_dataset", lambda: small)
    _render_all_pages()


def test_new_visit_starts_on_home_page(fresh_caches, monkeypatch):
    """Browsers reopen the last page's URL (restored tab, address completion): a new session must still start on
    page 1, not on page 2 (bug found after the deployment of 15 Nov 2026); later visits to that URL do open it."""
    small = synthetic.build_dataset(with_zoom=True)
    monkeypatch.setattr(data_prep, "load_default_dataset", lambda: small)
    at = AppTest.from_file(APP, default_timeout=180)
    at._page_hash = calc_hash("target")                   # first request of the session: the URL of page 2
    at.run()
    assert not at.exception and at.title[0].value == "AnalogExplorer"
    open_page(at, "target")                               # within the session, the URL is honoured
    assert not at.exception and at.title[0].value == "Target & Mission Configuration"


def test_app_stops_without_a_dataset(fresh_caches, monkeypatch):
    """No dataset on disk -> a clear error, never invented data."""
    monkeypatch.setattr(data_prep, "find_dataset_dir", lambda: None)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception and at.error and "--real" in at.error[0].value


def test_page2_target_and_mission_follow_on_every_page(fresh_caches, monkeypatch):
    """Page 1 presents, page 2 selects the target (body, then site) and the mission (Mission or Expert mode);
    every page uses them and a new choice starts again from the best site (15 Nov 2026)."""
    small = synthetic.build_dataset(with_zoom=True)
    monkeypatch.setattr(data_prep, "load_default_dataset", lambda: small)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.sidebar.selectbox                                  # choices live on page 2, not the sidebar
    open_page(at, "target")
    wanted, body = small.targets["target_id"].iloc[-1], small.targets["body"].iloc[-1]
    at.selectbox[0].set_value(body).run()                            # celestial body, then target site
    at.session_state["cfg"]["site_rank"] = 3
    at.selectbox[1].set_value(wanted)
    assert at.selectbox[2].label == "Mode" and at.selectbox[3].label == "Case study"
    at.selectbox[3].set_value("Solar Panels Deployment").run()
    assert not at.exception
    cfg = at.session_state["cfg"]
    assert cfg["target_id"] == wanted and cfg["use_case"] == "Solar Panels Deployment" and cfg["site_rank"] == 1
    name = small.targets["name"].iloc[-1]
    for page in PATHS[2:5]:                                  # the pages that show the context line
        open_page(at, page)
        assert not at.exception, page
        context = " ".join(str(m.value) for m in at.markdown)        # target and mission badges
        assert name in context and "Solar Panels Deployment" in context, page
    open_page(at, "target")
    at.selectbox[2].set_value("Expert Mode").run()                   # expert mode: one weight slider per dimension
    assert not at.exception and at.session_state["cfg"]["mode"] == "Expert Mode"
    assert any("weight" in str(s.label) for s in at.slider)


def test_robustness_box_takes_effect_at_once(fresh_caches, monkeypatch):
    """Page 3: the robustness box sits below the table it controls; unticking it must drop the column on the same
    rerun, not on the next one (bug found in the 15 Nov 2026 audit)."""
    small = synthetic.build_dataset(with_zoom=True)
    monkeypatch.setattr(data_prep, "load_default_dataset", lambda: small)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    open_page(at, "profile")
    assert "Robustness" in at.dataframe[0].value.columns
    at.checkbox(key="robustness_box").uncheck().run()
    assert not at.exception and "Robustness" not in at.dataframe[0].value.columns
    at.checkbox(key="robustness_box").check().run()
    assert "Robustness" in at.dataframe[0].value.columns


def test_new_deployed_version_clears_the_caches(fresh_caches, monkeypatch, tmp_path):
    """Streamlit Cloud reloads a new commit without restarting: objects cached by the old code must not outlive it
    (15 Nov 2026: 'Context' object has no attribute 'place' after the place names were deployed)."""
    small = synthetic.build_dataset(with_zoom=True)
    builds = []
    monkeypatch.setattr(data_prep, "load_default_dataset", lambda: builds.append(1) or small)
    (tmp_path / "meta.json").write_text("{}")
    monkeypatch.setattr(data_prep, "find_dataset_dir", lambda: tmp_path)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.run()
    assert not at.exception and len(builds) == 1                     # same version: the cache is kept
    (tmp_path / "places.csv").write_text("tile,place\n")             # a new version is deployed
    at.run()
    assert not at.exception and len(builds) == 2                     # caches cleared, context rebuilt
