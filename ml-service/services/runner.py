def get_test_cases(problem):
    problem = problem.lower()

    if "add" in problem or "sum" in problem:
        return [
            ((2, 3), 5),
            ((10, 20), 30),
            ((-1, 1), 0),
        ]

    elif "multiply" in problem or "product" in problem:
        return [
            ((2, 3), 6),
            ((10, 20), 200),
            ((-1, 5), -5),
        ]

    elif "subtract" in problem or "difference" in problem:
        return [
            ((5, 3), 2),
            ((20, 10), 10),
            ((-1, 1), -2),
        ]
    
    elif "prime" in problem:
        return [
            (([2, 3, 4, 5],), [2, 3, 5]),
            (([10, 11, 12, 13],), [11, 13]),
            (([1, 6, 7, 9],), [7]),
        ]

    return []


def run_code(code, problem):
    try:
        local_env = {}
        exec(code, local_env)

        functions = [v for v in local_env.values() if callable(v)]

        if not functions:
            return False, 0, "No function found"

        func = functions[0]
        tests = get_test_cases(problem)

        if not tests:
            return False, 0, "No test cases found"

        passed_count = 0
        total = len(tests)

        for inputs, expected in tests:
            try:
                result = func(*inputs)
                if result == expected:
                    passed_count += 1
            except:
                continue

        score = passed_count / total

        if score == 1:
            return True, score, "All tests passed"

        return False, score, f"{passed_count}/{total} tests passed"

    except Exception as e:
        return False, 0, str(e)