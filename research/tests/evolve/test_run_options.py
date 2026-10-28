from pathlib import Path

import numpy as np
import pytest

from laplace_research.evolve.islands import read_islands
from laplace_research.evolve.panel import Panel
from laplace_research.evolve.run import Run


def test_island_hypotheses_come_from_a_toml_file_in_its_order(tmp_path: Path) -> None:
    path = tmp_path / "islands.toml"
    path.write_text('tails = """\nFat tails\n    after drops."""\nclock = "The hour of day."\n')
    assert read_islands(path) == {"tails": "Fat tails after drops.", "clock": "The hour of day."}
    assert list(read_islands(path)) == ["tails", "clock"]


@pytest.mark.parametrize(
    "text", ["", "tails = 3\n", 'tails = "  "\n', '[tails]\ntext = "x"\n'], ids=["empty", "number", "blank", "table"]
)
def test_an_island_file_needs_one_directive_per_name(tmp_path: Path, text: str) -> None:
    path = tmp_path / "islands.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="one non-empty directive"):
        read_islands(path)


def test_init_seeds_the_given_islands_and_samples_the_search_at_the_given_step(panel: Panel, tmp_path: Path) -> None:
    islands = {"tails": "Fat tails after drops.", "clock": "The hour of day."}
    run = Run.init(tmp_path / "evolve-t", tmp_path / "cache", panel=panel, islands=islands, search_step=7)
    assert [(row["name"], row["directive"]) for row in run.db.islands()] == list(islands.items())
    assert run.config.search[2] == 7
    assert set(np.diff(run.search.times).tolist()) == {7}
    assert Run(run.dir).config == run.config  # the step is part of the stored config


def test_init_refuses_no_islands_and_a_step_below_one_hour(panel: Panel, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one island"):
        Run.init(tmp_path / "evolve-a", tmp_path / "cache", panel=panel, islands={})
    with pytest.raises(ValueError, match="at least 1 hour"):
        Run.init(tmp_path / "evolve-b", tmp_path / "cache", panel=panel, search_step=0)
