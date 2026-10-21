import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("atom_edc", Path(__file__).parents[1] / "scripts" / "atom_edc.py")
atom_edc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atom_edc)


def test_to_triples_drops_time_dedupes_and_counts_timed_quintuples():
    quints = [("Herbert", "is_born_in", "Mülhausen", ["1904-12-15"], []),
              ("Herbert", "is_born_in", "Mülhausen", [], []),
              ("Herbert", "is_diplomat_of", "Germany", [], ["1991-08-10"]),
              ("Herbert", "dies_in", "Badenweiler", [], [])]
    triples, timed = atom_edc.to_triples(quints)
    assert triples == [["Herbert", "is_born_in", "Mülhausen"], ["Herbert", "is_diplomat_of", "Germany"],
                       ["Herbert", "dies_in", "Badenweiler"]]
    assert timed == 2
    with_time, _ = atom_edc.to_triples(quints, with_time=True)
    assert with_time == [["Herbert", "is_born_in", "Mülhausen"], ["Herbert", "is_born_in (start time)", "1904-12-15"],
                         ["Herbert", "is_diplomat_of", "Germany"], ["Herbert", "is_diplomat_of (end time)", "1991-08-10"],
                         ["Herbert", "dies_in", "Badenweiler"]]


def test_factoid_list_skips_blank_and_failed_outputs():
    class Out:
        atomic_fact = ["Trane is located in Swords.", "  ", "Trane was founded in 1913."]

    assert atom_edc.factoid_list(Out()) == ["Trane is located in Swords.", "Trane was founded in 1913."]
    assert atom_edc.factoid_list(ValueError("bad json")) == []
    assert atom_edc.factoid_list(None) == []
