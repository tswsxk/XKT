# coding: utf-8
# 2021/8/24 @ tongshiwei

__all__ = ["pseudo_data_generation"]


def pseudo_data_generation(ku_num):
    import random
    random.seed(10)

    raw_data = [
        [
            (random.randint(0, ku_num - 1), random.randint(-1, 1))
            for _ in range(random.randint(2, 20))
        ] for _ in range(100)
    ]

    return raw_data
