"""Candidate values read from a request (s1agent.candidates)."""
from s1agent import candidates as C


def test_paths():
    t = "Segment /data/ct/p7.nii.gz, save to ~/out/seg and read ./cfg.json or results.csv."
    assert C.paths(t) == ["/data/ct/p7.nii.gz", "~/out/seg", "./cfg.json", "results.csv"]


def test_numbers_skip_paths_and_identifiers():
    t = "Train liver_v1 for 100 epochs, 5-fold, 3d_fullres, 600 dpi, lr 0.0001 on /data/ds501/img2 (10%)."
    assert C.numbers(t) == ["100", "5", "600", "0.0001", "10"]


def test_number_forms():
    assert C.numbers("use nine classes, a batch size of 6? and lr 2e-4, num_classes=`8`") == ["6", "2e-4", "8", "9"]
    assert C.number_words("ten, twenty-five and fifty; forty-two and ten; two hundred and fifty") == \
        ["10", "25", "50", "42", "10", "250"]
    assert C.occurrences("gamma 2 and 2 classes, two heads", "2") == 3


def test_quoted_and_identifiers():
    t = "Target 'grade', I'm sure the model's 'weights.pt.' are fine"
    assert C.quoted(t) == ["grade", "weights.pt."]                 # apostrophes inside words are not quotes
    assert C.quoted("Target 'grade.'", verbatim=False) == ["grade"]
    t = "Target 'grade', check job_20261027_1a2b with model kits_pretrained (Dataset220_KiTS)"
    assert C.identifiers(t) == ["job_20261027_1a2b", "kits_pretrained", "Dataset220_KiTS"]


def test_phrases_split_lists():
    assert C.phrases("Segment the liver, kidneys and spleen in /data/ct.nii.gz") == ["liver", "kidneys", "spleen"]
    assert C.phrases("keep liver + spleen please") == ["liver", "spleen"]


def test_csv_columns(tmp_path):
    f = tmp_path / "d.csv"
    f.write_text("patient_id,age,outcome\n1,50,0\n")
    assert C.csv_columns(f) == ["patient_id", "age", "outcome"]
    assert C.csv_columns(tmp_path / "missing.csv") == []


def test_folders_and_joined_paths():
    assert C.stated_folders("put it in the same folder as the CSV", ["/data/a/x.csv"]) == ["/data/a"]
    assert C.stated_folders("use /data/idh/train.npz", ["/data/idh/train.npz"]) == ["/data/idh"]
    assert C.joined_paths("load best.pt from /models/run3 and write a folder called merlin_out next to it") == \
        ["/models/run3/best.pt", "/models/best.pt", "/models/run3/merlin_out", "/models/merlin_out"]
    assert C.quoted_paths("open '/home/j/Shared Data/scan.nii.gz' now") == ["/home/j/Shared Data/scan.nii.gz"]
    assert C.bare_files("load heart_norm.h5ad and os.status") == ["heart_norm.h5ad"]
    assert C.renamed_outputs("convert /d/Y_KO_14.raw to mzML with the same stem", False) == ["/d/Y_KO_14.mzML"]


def test_numbers_in_other_forms():
    assert C.derived_numbers("binary task", "num_classes", {"type": "integer"}) == ["2"]
    assert C.derived_numbers("20 percent held out, one fifth", "test_size",
                             {"type": "number", "description": "Fraction of the data"}) == ["0.2"]
    assert C.derived_numbers("wait 5 minutes", "timeout_seconds", {"type": "integer"}) == ["300"]
    assert C.durations("half a minute or 4 min 30 s", "timeout_seconds", {}) == ["30", "270"]
    assert C.range_items("folds 0 through 4") == ["0", "1", "2", "3", "4"]
    assert C.class_counts("glioma, meningioma, pituitary and no tumour; grade 0 to 4") == ["4", "2", "5"]
    assert C.number_groups("position 43,044,295 and 50k reads, alpha .1") == ["43044295", "50000", "0.1"]
    assert C.number_scales("2 million reads, 95th percentile, 0.1/0.25/0.5") == ["2000000", "95", "0.1", "0.25", "0.5"]
    assert C.split_fractions("a 70/30 split") == ["0.7", "0.3"]
    assert C.complement_levels("90% prediction intervals", {"description": "1-alpha coverage"}) == ["0.1"]


def test_dictionaries_and_items():
    assert C.dicts('use {"rotation": 20, "flip": true} and weights 0:1.0, 1:9.0') == \
        [{"rotation": 20, "flip": True}, {"0": 1.0, "1": 9.0}]
    assert C.prose_dicts("1.0 for class 0 and 1.6 for class 1") == [{"0": 1.0, "1": 1.6}]
    assert C.written_items("Class names: no DR, mild, moderate. Also cyst, normal, stone and tumor") == \
        ["no DR", "mild", "moderate", "DR", "cyst", "normal", "stone", "tumor"]
    assert C.grouped_lines("T cells: CD3D, CD3E\nB cells: CD19, MS4A1") == [{"T cells": ["CD3D", "CD3E"],
                                                                           "B cells": ["CD19", "MS4A1"]}]
    assert C.bracket_lists("[chr1, 100, 200, a] and [chr2, 300, 400, b]")[0] == [["chr1", 100, 200, "a"],
                                                                                 ["chr2", 300, 400, "b"]]
    assert C.written_arrays("[[94.065, 35], [146.06, 100]]") == [[[94.065, 35], [146.06, 100]]]


def test_genomic_and_spectral_notation():
    assert C.chromosome_names("chromosome 7 and 19:44908822:C:T") == ["chr7", "chr19"]
    assert C.variant_alleles("19:44908822:C:T and reference allele G") == ["C", "T", "G"]
    assert C.alleles("chr7:140753336 A>T") == ["A", "T"]
    assert C.strand_marks("on + and - strands, minus strand, unstranded") == ["+", "-", "."]
    assert C.span_ends("a 2 kb interval centred on chr17:43,044,295") == ["43043295", "43045295"]
    assert C.length_units("2 kb up, from 27.0 to 28.0 Mb") == ["2000", "28000000", "27000000"]
    assert C.joined_tokens("L858R,T790M with MS²PIP v2.0 and b/y ions, Carbamidomethyl (C)") == \
        ["L858R,T790M", "MS2PIP", "v2.0", "Carbamidomethyl (C)", "by"]
    assert C.milli_units("10 mDa tolerance") == (["0.01"], ["Da"])
    assert C.charge_forms("3+ ions, doubly charged") == ["3", "2"]
    assert C.compact_ids("tissue UBERON:0002048 and num_classes=5 and plan+preprocess") == ["UBERON:0002048"]


def test_search_expressions():
    assert C.boolean_queries("search long COVID AND fatigue") == ["long COVID AND fatigue"]
    assert C.bracket_queries("(psilocybin OR MDMA) AND depression") == ["(psilocybin OR MDMA) AND depression"]
    assert C.search_chunks("trials for type 1 diabetes in children") == ["type 1 diabetes", "type 1"]
    assert C.possessive_phrases("trials in Crohn's disease") == ["Crohn's disease"]
    assert C.colon_blocks("run: SELECT a, COUNT(*) FROM t", "A read-only SELECT statement") == ["SELECT a, COUNT(*) FROM t"]
