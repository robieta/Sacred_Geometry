# -*- coding: utf-8 -*-

import collections
import cProfile
import functools
import math
import operator
import os
import shutil
import sys
import timeit
import typing


assert sys.version_info[:2] >= (3, 7)


DICE_FACES = 8
LRU_CACHE_SIZE = int(1e6)
MAX_SUBSEGMENT_FIRST_PASS = 110  # Drop large values as they are generally not needed and exacerbate the combinatorics
SUBSOLUTION_CACHE = {}

# These are used for rendering formulas in human readable formats.
MODE_FORMATS = ["{a} + {b}", "{a} - {b}", "{b} - {a}", "{a} * {b}", "{a} / {b}", "{b} / {a}"]
MODE_PRIORITY = [0, 1, 1, 2, 3, 3]
MAX_PRIORITY = max(MODE_PRIORITY) + 1

TARGETS = [(3, 5, 7), (11, 13, 17), (19, 23, 29), (31, 37, 41), (43, 47, 53), (59, 61, 67), (71, 73, 79),
           (83, 89, 97), (101, 103, 107)]
TARGET_SETS = [set(i) for i in TARGETS]
OUTPUT_DIR = os.path.join(os.getcwd(), f"results_{DICE_FACES}_faces")


@functools.lru_cache(maxsize=100)
def fact(x):
    # type: (int) -> int
    """Wrap math.factorial in a cache since it will be repeatedly called for small values.
    """
    return math.factorial(x)


def vector_to_indexed_counts(x):
    # type: (typing.Iterable) -> (typing.Tuple[int], typing.Tuple[int])
    """Convert a standard vector of integers to a compact and representation.
    """
    count_dict = collections.defaultdict(int)
    for i in x:
        count_dict[i] += 1

    keys = sorted(count_dict.keys())
    return tuple(keys), tuple(count_dict[i] for i in keys)


@functools.lru_cache(maxsize=LRU_CACHE_SIZE)
def partition_counts(counts):
    # type: (tuple) -> typing.List[tuple]
    """Returns all possible partitions of a vector.

    The inputs are expected to be in compact form (see vector_to_indexed_counts) and will return a list of
    partitions. (conjugates are implied)
    """
    if len(counts) == 0:
        return [()]

    subsequence = partition_counts(counts[1:])

    output = []
    for i in range(counts[0] + 1):
        new_entries = [(i,) + seq for seq in subsequence]
        new_entries = [i for i in new_entries if sum(i)]
        output.extend(new_entries)

    return output or [counts]


@functools.lru_cache(maxsize=LRU_CACHE_SIZE)
def group_contiguous(x):
    # type: (tuple) -> typing.List[tuple]
    """Separate tuple into blocks of contiguous elements.
    """
    x_grouped = []
    for i in sorted(x):
        if not x_grouped:
            x_grouped.append([i])
        elif i == x_grouped[-1][-1] + 1:
            x_grouped[-1].append(i)
        else:
            x_grouped.append([i])
    return [tuple(i) for i in x_grouped]


@functools.lru_cache(maxsize=LRU_CACHE_SIZE)
def int_combine(a, b):
    # type: (int, int) -> typing.Tuple[typing.Tuple[int, int], ...]
    """Enumerate all legal combinations of two values. (either raw values or subgroups)
    """
    output = [a + b, a - b, b - a, a * b, None if (b == 0 or a % b) else int(a // b),
              None if (a == 0 or b % a) else int(b // a)]
    return tuple((k, mode) for mode, k in enumerate(output) if k is not None)


@functools.lru_cache(maxsize=LRU_CACHE_SIZE)
def _vector_combine(x, y, max_subsegment):
    results = {}
    for i in x:
        for j in y:
            if max_subsegment is not None:
                results.update({k: (i, j, mode) for k, mode in int_combine(i, j) if (0 <= k <= max_subsegment)})
            else:
                results.update({k: (i, j, mode) for k, mode in int_combine(i, j)})

    return results.items()


@functools.lru_cache(maxsize=LRU_CACHE_SIZE)
def vector_combine(x, y, max_subsegment):
    x_grouped = group_contiguous(x)
    y_grouped = group_contiguous(y)

    results = {}
    for x_chunk in x_grouped:
        for y_chunk in y_grouped:
            results.update({k: v for k, v in _vector_combine(x_chunk, y_chunk, max_subsegment)})
            if max_subsegment is not None and len(results) == max_subsegment + 1:
                return results
    return results


@functools.lru_cache(maxsize=LRU_CACHE_SIZE)
def conj_vector(partition, counts):
    return tuple(ct - partition[i] for i, ct in enumerate(counts))


@functools.lru_cache(maxsize=LRU_CACHE_SIZE)
def all_computable(keys, counts, max_subsegment):
    if sum(counts) == 1:
        return tuple(keys[i] for i, ct in enumerate(counts) if ct)

    partitions = partition_counts(counts)
    seen = set()
    out_set = set()
    for partition in partitions:
        conjugate = conj_vector(partition, counts)
        if conjugate in seen or not sum(conjugate):
            continue
        seen.update([partition, conjugate])

        partition_computable = all_computable(keys, partition, max_subsegment)
        conjugate_computable = all_computable(keys, conjugate, max_subsegment)

        combined_results = vector_combine(
            partition_computable, conjugate_computable, max_subsegment
        )
        for k, v in combined_results.items():
            SUBSOLUTION_CACHE[(k, counts, keys)] = (partition, conjugate, v)
        combined_computable = combined_results.keys()

        out_set.update(combined_computable)
        if max_subsegment is not None and len(out_set) == max_subsegment + 1:
            break  # all values are computable

    return tuple(sorted(out_set))


@functools.lru_cache(maxsize=LRU_CACHE_SIZE)
def code_op_to_str(args):
    if isinstance(args, (int, str)):
        return args, MAX_PRIORITY
    elif len(args) == 1:
        return args[0], MAX_PRIORITY
    a, b, mode = args
    [a, a_priority], [b, b_priority] = code_op_to_str(a), code_op_to_str(b)
    if a_priority <= MODE_PRIORITY[mode]:
        a = f"({a})"
    if b_priority <= MODE_PRIORITY[mode]:
        b = f"({b})"

    return MODE_FORMATS[mode].format(a=a, b=b),  MODE_PRIORITY[mode]


@functools.lru_cache(maxsize=LRU_CACHE_SIZE)
def _lookup_formula(cache_key):
    (_, counts, keys) = cache_key
    if sum(counts) == 1:
        return tuple(keys[i] for i, ct in enumerate(counts) if ct)[0]

    partition, conjugate, [partition_target, conjugate_target, mode] = SUBSOLUTION_CACHE[cache_key]
    cache_key0 = (partition_target, partition, keys)
    cache_key1 = (conjugate_target, conjugate, keys)
    return _lookup_formula(cache_key0), _lookup_formula(cache_key1), mode


def lookup_formula(keys, counts, target):
    if sum(counts) == 1:
        return str(target) if tuple(keys[i] for i, ct in enumerate(counts) if ct)[0] == target else None

    cache_key = (target, counts, keys)
    if cache_key not in SUBSOLUTION_CACHE:
        return None

    result_str = code_op_to_str(_lookup_formula(cache_key))[0]
    result_str_eval = eval(result_str)

    assert int(result_str_eval) == result_str_eval
    assert int(result_str_eval) == target

    return code_op_to_str(_lookup_formula(cache_key))[0]


def generate_all_rolls(max_num_dice):
    output = []
    rolls = [(i,) for i in range(1, DICE_FACES + 1)]
    output.append(rolls)
    for i in range(max_num_dice - 1):
        rolls_new = []
        for subroll in rolls:
            for j in range(1, DICE_FACES + 1):
                if j < subroll[-1]:
                    continue
                rolls_new.append(subroll + (j,))
        rolls = rolls_new
        output.append(rolls)

    return output


def get_formulas(roll, max_subsegment):
    keys, counts = vector_to_indexed_counts(roll)

    omega_denom = functools.reduce(operator.mul, [fact(i) for i in counts], 1)
    assert not fact(sum(counts)) % omega_denom
    omega = fact(sum(counts)) // omega_denom
    x = {i for i in all_computable(keys, counts, max_subsegment) if i > 0}
    targets = [i.intersection(x) for i in TARGET_SETS]
    targets = [i.pop() if i else None for i in targets]

    return [f"{lookup_formula(keys, counts, i)} = {i}" if i else "" for i in targets], omega


CACHED_FNS = [partition_counts, int_combine, group_contiguous, _vector_combine, vector_combine, all_computable,
              conj_vector, fact, code_op_to_str, _lookup_formula]
def clear_caches():
    for fn in CACHED_FNS:
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()


def cache_info():
    name_len = max([len(fn.__name__) for fn in CACHED_FNS])
    for fn in CACHED_FNS:
        if not hasattr(fn, "cache_info"):
            continue
        info = fn.cache_info()
        hit_rate = info.hits / (info.hits + info.misses)
        print(fn.__name__.ljust(name_len + 3), f"hit_rate={hit_rate*100:.1f}%  ", info)


def main(max_num_dice=4):
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    summary_file = os.path.join(OUTPUT_DIR, "summary.txt")

    st = timeit.default_timer()
    compute_time = 0
    overshoot_solutions = []
    for i, rolls in enumerate(generate_all_rolls(max_num_dice)):
        loop_st = timeit.default_timer()
        num_dice = i + 1
        results = []
        omegas = []
        for roll in rolls:
            result, omega = get_formulas(roll, MAX_SUBSEGMENT_FIRST_PASS)
            omegas.append(omega)

            # A more efficient ordering would be to do all of the rigorous passes at the end to reduce cache eviction,
            # but this is easier to read.
            if not all(result):
                prior_result = result.copy()
                result, _ = get_formulas(roll, None)
                overshoot_solutions.extend([r for i, r in enumerate(result) if r and not prior_result[i]])
            results.append(result)
        compute_time += timeit.default_timer() - loop_st
        results = list(zip(*results))
        widths = [max([len(r) for r in result]) for result in results]
        results = list(zip(*results))

        perfect_count = 0
        omega_len = max([len(str(w)) for w in omegas])
        assert sum(omegas) == DICE_FACES ** num_dice
        lines, impossible_lines = [], []
        for roll, omega, result in zip(rolls, omegas, results):
            perfect_count += all(result)
            omega_str = f"(ω = {omega})".ljust(omega_len + 7)
            formulas = "   |   ".join([r.replace("=", "{}=").format(" " * (widths[i] - len(r)))
                                                           if r else " " * widths[i] for i, r in enumerate(result)])

            lines.append(f"{list(roll)}  {omega_str}    {formulas}")
            if not all(result):
                impossible_str = " ".join([" " if r else str(i+1) for i, r in enumerate(result)])
                impossible_lines.append(f"{list(roll)}  {omega_str}    {impossible_str}")

        omega_perfect = sum([omega if all(result) else 0 for omega, result in zip(omegas, results)])

        table_path = os.path.join(OUTPUT_DIR, f"solutions_{num_dice}_dice.txt")

        with open(table_path, "wt", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        if impossible_lines:
            impossible_path = os.path.join(OUTPUT_DIR, f"impossible_{num_dice}_dice.txt")
            with open(impossible_path, "wt", encoding="utf-8") as f:
                for line in impossible_lines:
                    f.write(line + "\n")

        summary = [
            f"{num_dice} dice complete:",
            f"  {timeit.default_timer() - st:.4f} seconds  ({compute_time:.4f} in primary compute)",
            f"  {perfect_count} / {len(results)} rolls succeed at all 9 levels.",
            f"  {omega_perfect} / {sum(omegas)}  ({omega_perfect / sum(omegas) * 100:.9f}%) that a roll will succeed "
            f"at all 9 levels",
            "",
        ]
        with open(summary_file, "at", encoding="utf-8") as f:
            f.write("\n".join(summary))
            f.write("\n")
        print("\n".join(summary))

    with open(os.path.join(OUTPUT_DIR, "honorable_mentions.txt"), "wt", encoding="utf-8") as f:
        f.write("Overshoot:\n")
        f.write(f"Cases where an intermediate result > {MAX_SUBSEGMENT_FIRST_PASS} is needed to hit a given level.\n")
        f.write("\n".join(overshoot_solutions))


if __name__ == "__main__":
    # cProfile.run("main(12)")
    main(20)
    cache_info()
