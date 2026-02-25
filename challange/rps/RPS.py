def player(prev_play, opponent_history=[], play_order={}):
    if not prev_play:
        opponent_history.clear()
        play_order.clear()
        return "R"

    opponent_history.append(prev_play)

    n = 3
    if len(opponent_history) > n:
        pattern = "".join(opponent_history[-(n + 1) : -1])
        if pattern not in play_order:
            play_order[pattern] = {"R": 0, "P": 0, "S": 0}
        play_order[pattern][prev_play] += 1

    last_n = "".join(opponent_history[-n:])

    prediction = "R"
    if last_n in play_order:
        prediction = max(play_order[last_n], key=play_order[last_n].get)
    elif len(opponent_history) > 0:
        prediction = opponent_history[-1]

    counter = {"R": "P", "P": "S", "S": "R"}
    return counter[prediction]
