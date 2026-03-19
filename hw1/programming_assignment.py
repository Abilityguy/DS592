import random
from typing import Dict, Tuple

# We can write an adjacency list representation of the network using a dictionary.
nn = {
    "11": ("21", "22"),
    "12": ("22", "23"),
    "21": ("31", "32"),
    "22": ("31", "32"),
    "23": ("31", "32"),
    "31": ("41", "42", "43"),
    "32": ("41", "42", "43"),
    "41": (),  # Empty tuple indicates final node
    "42": (),
    "43": (),
}

FINAL_NODES = ("41", "42", "43")
IDEAL = (1 / 3, 1 / 3, 1 / 3)


def get_random_next_node(current_node: str) -> str:
    """
    Given the current node, returns a randomly selected next node from its neighbors.

    :param current_node: The identifier of the current node.
    :type current_node: str
    :return: The identifier of the next node.
    :rtype: str
    """
    neighbors = nn.get(current_node, ())
    if not neighbors:
        return None  # No next node available
    return random.choice(neighbors)


def random_walk(probs: dict) -> str:
    """
    Simulates a random walk through the network.

    :param probs: A dictionary where keys are node identifiers and values are probabilities of starting at that node.
    :type probs: dict

    :return: The identifier of the final node reached in the random walk.
    :rtype: str
    """
    nodes = list(probs.keys())
    probabilities = list(probs.values())

    starting_node = random.choices(nodes, weights=probabilities, k=1)[0]
    current_node = starting_node

    while True:
        next_node = get_random_next_node(current_node)
        if next_node is None:
            break
        current_node = next_node

    return current_node


def simulate_walks(n: int, probs: dict) -> dict:
    """
    Simulates multiple random walks and counts the occurrences of final nodes.

    :param n: The number of random walks to simulate.
    :type n: int
    :param probs: A dictionary where keys are node identifiers and values are probabilities of starting at that node.
    :type probs: dict

    :return: A dictionary with final node identifiers as keys and their occurrence counts as values.
    :rtype: dict
    """
    counter = {}
    for _ in range(n):
        final_node = random_walk(probs)
        if final_node in counter:
            counter[final_node] += 1
        else:
            counter[final_node] = 1
    return counter


def counter_to_probs(counter: Dict[str, int], n: int) -> Tuple[float, float, float]:
    return tuple(counter.get(node, 0) / n for node in FINAL_NODES)


def get_norm(x: Tuple[float, float, float]) -> float:
    return sum(v * v for v in x) ** 0.5


def fmt_triplet(x: Tuple[float, float, float], digits: int = 4) -> str:
    return "(" + ", ".join(f"{v:.{digits}f}" for v in x) + ")"


def fmt_probs_dict(d: Dict[str, float]) -> str:
    # consistent ordering
    items = [(k, d[k]) for k in sorted(d.keys())]
    return "{" + ", ".join(f"'{k}': {v:g}" for k, v in items) + "}"


def print_table(rows):
    headers = ["n", "probs", "empirical pi_4", "||empirical pi_4 - ideal||_2"]
    # compute column widths
    widths = [len(h) for h in headers]
    for r in rows:
        for i, h in enumerate(headers):
            widths[i] = max(widths[i], len(str(r[h])))

    # print header
    sep = " | "
    header_line = sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_line)
    print(rule_line)

    # print rows
    for r in rows:
        line = sep.join(str(r[h]).ljust(widths[i]) for i, h in enumerate(headers))
        print(line)


def main():
    n_vals = (10, 100, 1000, 10000, 100000)
    probs_list = [
        {"11": 0.5, "12": 0.5},
        {"11": 0.4, "12": 0.6},
        {"11": 0.1, "12": 0.9},
    ]

    rows = []
    for probs in probs_list:
        for n in n_vals:
            counter = simulate_walks(n, probs)
            emp = counter_to_probs(counter, n)
            diff = tuple(emp[i] - IDEAL[i] for i in range(3))

            rows.append(
                {
                    "n": n,
                    "probs": fmt_probs_dict(probs),
                    "empirical pi_4": fmt_triplet(emp, digits=4),
                    "||empirical pi_4 - ideal||_2": f"{get_norm(diff):.6f}",
                }
            )

    print_table(rows)


if __name__ == "__main__":
    main()
