from typing import Iterable

ID_TO_AA = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
}


def ids_to_sequence(token_ids: Iterable[int]) -> str:
    return "".join(ID_TO_AA.get(int(idx), "X") for idx in token_ids)


def write_fasta(path: str, header: str, sequence: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f">{header}\n")
        f.write(sequence + "\n")
