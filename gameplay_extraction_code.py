def scrape_gameplay_files_plays(file_name):
  # scrape through gameplay log under file_name for all card-playing moves made during the game
    card_pattern = re.compile(r"Card played: Rank (\d+), Color (\w+)")
    action_pattern = re.compile(r"Agent: (\d+) action: (\{[^}]+\})\.")
    results = []
    pending_card = None
    with open(file_name, 'r') as f:
        for line in f:
            card_match = card_pattern.search(line)
            if card_match:
                pending_card = {'rank': int(card_match.group(1)), 'color': card_match.group(2)}
                continue
            if pending_card is not None:
                action_match = action_pattern.search(line)
                if action_match:
                    results.append({
                        'agent': int(action_match.group(1)),
                        'color': pending_card['color'],
                        'rank': pending_card['rank'],
                    })
                pending_card = None
    return results


def plt_weight_against_mean_plays(file_name, weights, label):
    # for all files that include file_name in their name
    # return the total number of cards played by each player for each coloiur across all games
    files_gp = sorted([f for f in os.listdir('logs/gameplay/') if f.startswith(file_name)])
    colors = ["R", "Y", "G", "W", "B"]
    print(np.shape(files_gp))
    total_counts = [{c: 0 for c in colors} for _ in range(5)]
    print(np.shape(total_counts))
    print(total_counts)
    for file in files_gp:
        results = scrape_gameplay_files_plays('logs/gameplay/' + file)
        for res in results:
            total_counts[res['agent']][res['color']] += 1
    return total_counts

def scrape_gameplay_files_hints(file_name):
    # scrape through gameplay log under file_name for all hint-giving moves made during the game
    action_pattern = re.compile(r"Agent: (\d+) action: (\{[^}]+\})\. (?:Target player: (\d+))?")
    card_pattern = re.compile(r"^([RYGWB])(\d) \|\|")
    results = []

    with open(file_name, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        match = action_pattern.search(lines[i])
        if not match:
            i += 1
            continue

        agent = int(match.group(1))
        action = ast.literal_eval(match.group(2))
        target_player = int(match.group(3)) if match.group(3) else None

        if action['action_type'] == 'REVEAL_COLOR':
            results.append({'agent': agent, 'color': action['color'], 'target_player': target_player})
            i += 1

        elif action['action_type'] == 'REVEAL_RANK':
            hinted_rank = action['rank'] + 1  # log is 1-indexed

            j = i + 1
            while j < len(lines) and 'Hands:' not in lines[j]:
                j += 1
            j += 1  # skip "Hands:" line

            hands = {}
            p, hand = 0, []
            while j < len(lines) and 'Deck size:' not in lines[j]:
                s = lines[j].strip()
                if s == '-----':
                    hands[p] = hand
                    p, hand = p + 1, []
                elif s not in ('Cur player', ''):
                    m = card_pattern.match(s)
                    if m:
                        hand.append((m.group(1), int(m.group(2))))
                j += 1
            hands[p] = hand

            if target_player in hands:
                for color, rank in hands[target_player]:
                    if rank == hinted_rank:
                        results.append({'agent': agent, 'color': color, 'target_player': target_player})
            i = j

        else:
            i += 1

    return results

def plt_weight_against_mean_hints(file_name, weights, label): 
    # for all files that include file_name in their name
    # return the total number of hints given by each player for each coloiur across all games
    files_gp = sorted([f for f in os.listdir('logs/gameplay/') if f.startswith(file_name)])
    colors = ["R", "Y", "G", "W", "B"]
    print(np.shape(files_gp))
    total_counts = [{c: 0 for c in colors} for _ in range(5)]
    print(np.shape(total_counts))
    for file in files_gp:
        results = scrape_gameplay_files_hints('logs/gameplay/' + file)
        for res in results:
            total_counts[res['agent']][res['color']] += 1
    return total_counts

def plt_weight_against_mean_received_hints(file_name, weights, label):
    # for all files that include file_name in their name
    # return the total number of hints received by each player for each coloiur across all games
    files_gp = sorted([f for f in os.listdir('logs/gameplay/') if f.startswith(file_name)])
    colors = ["R", "Y", "G", "W", "B"]
    print(np.shape(files_gp))
    total_counts = [{c: 0 for c in colors} for _ in range(5)]
    print(np.shape(total_counts))
    for file in files_gp:
        results = scrape_gameplay_files_hints('logs/gameplay/' + file)
        for res in results:
            total_counts[res['target_player']][res['color']] += 1
    return total_counts
