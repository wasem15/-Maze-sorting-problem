import sys
import heapq
from typing import List, Tuple, Iterable, Dict

HALL_LEN = 11
DOORS = (2, 4, 6, 8)
HALL_STOPS = tuple(i for i in range(HALL_LEN) if i not in DOORS)
ENERGY = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
TARGET_ROOM = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

State = Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]]

def parse_input(lines: List[str]) -> State:
    hallway = tuple(lines[1][1:-1]) if len(lines) > 1 and len(lines[1]) >= 13 else tuple('.' * HALL_LEN)
    room_rows = [''.join(ch for ch in s if ch in 'ABCD') for s in lines if ''.join(ch for ch in s if ch in 'ABCD')]
    depth = len(room_rows)
    rooms = [tuple(room_rows[d][r] for d in range(depth)) for r in range(4)]
    return (tuple(hallway), (rooms[0], rooms[1], rooms[2], rooms[3]))

def goal_state(depth: int) -> State:
    hallway = tuple('.' for _ in range(HALL_LEN))
    rooms = tuple(tuple(chr(ord('A') + i) for _ in range(depth)) for i in range(4))
    return (hallway, rooms)

def hallway_path_clear(h: Tuple[str, ...], a: int, b: int) -> bool:
    if a < b:
        rng = range(a + 1, b + 1)
    else:
        rng = range(b, a)
    for i in rng:
        if i == a:
            continue
        if h[i] != '.':
            return False
    return True

def can_enter_room(room: Tuple[str, ...], amph: str) -> bool:
    return all((c == '.' or c == amph) for c in room)

def deepest_available(room: Tuple[str, ...]) -> int:
    for i in range(len(room) - 1, -1, -1):
        if room[i] == '.':
            return i
    return -1

def first_occupant(room: Tuple[str, ...]) -> int:
    for i, c in enumerate(room):
        if c != '.':
            return i
    return -1

def moves_from_state(state: State) -> Iterable[Tuple[int, State]]:
    hallway, rooms = state
    depth = len(rooms[0])

    for hpos, amph in enumerate(hallway):
        if amph == '.':
            continue
        r = TARGET_ROOM[amph]
        door = DOORS[r]
        if not hallway_path_clear(hallway, hpos, door):
            continue
        room = rooms[r]
        if not can_enter_room(room, amph):
            continue
        dpos = deepest_available(room)
        if dpos < 0:
            continue
        steps = abs(hpos - door) + (dpos + 1)
        cost = steps * ENERGY[amph]
        new_h = list(hallway)
        new_h[hpos] = '.'
        new_room = list(room)
        new_room[dpos] = amph
        new_rooms = list(rooms)
        new_rooms[r] = tuple(new_room)
        yield cost, (tuple(new_h), (new_rooms[0], new_rooms[1], new_rooms[2], new_rooms[3]))

    for r_index, room in enumerate(rooms):
        target = chr(ord('A') + r_index)
        if all(c in ('.', target) for c in room) and any(c == target for c in room):
            if not any(c not in ('.', target) for c in room):
                continue
        top_idx = first_occupant(room)
        if top_idx == -1:
            continue
        amph = room[top_idx]
        if all(c == target for c in room[top_idx:]):
            continue
        door = DOORS[r_index]

        left = door - 1
        while left >= 0 and hallway[left] == '.':
            if left in HALL_STOPS:
                steps = (top_idx + 1) + abs(left - door)
                cost = steps * ENERGY[amph]
                new_h = list(hallway)
                new_h[left] = amph
                new_room = list(room)
                new_room[top_idx] = '.'
                new_rooms = list(rooms)
                new_rooms[r_index] = tuple(new_room)
                yield cost, (tuple(new_h), (new_rooms[0], new_rooms[1], new_rooms[2], new_rooms[3]))
            left -= 1

        right = door + 1
        while right < HALL_LEN and hallway[right] == '.':
            if right in HALL_STOPS:
                steps = (top_idx + 1) + abs(right - door)
                cost = steps * ENERGY[amph]
                new_h = list(hallway)
                new_h[right] = amph
                new_room = list(room)
                new_room[top_idx] = '.'
                new_rooms = list(rooms)
                new_rooms[r_index] = tuple(new_room)
                yield cost, (tuple(new_h), (new_rooms[0], new_rooms[1], new_rooms[2], new_rooms[3]))
            right += 1

def dijkstra(start: State, goal: State) -> int:
    pq: List[Tuple[int, State]] = [(0, start)]
    best: Dict[State, int] = {start: 0}

    while pq:
        cost, state = heapq.heappop(pq)
        if state == goal:
            return cost
        if cost != best.get(state, 1 << 60):
            continue
        for move_cost, nxt in moves_from_state(state):
            nc = cost + move_cost
            if nc < best.get(nxt, 1 << 60):
                best[nxt] = nc
                heapq.heappush(pq, (nc, nxt))
    return -1

def solve(lines: List[str]) -> int:
    start = parse_input(lines)
    depth = len(start[1][0])
    return dijkstra(start, goal_state(depth))

def main():
    lines = [line.rstrip('\n') for line in sys.stdin]
    print(solve(lines))

if __name__ == "__main__":
    main()
