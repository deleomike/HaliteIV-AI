from kaggle_environments.envs.halite.helpers import *
from sklearn import preprocessing

# Directions a ship can move
directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]

le = preprocessing.LabelEncoder()
label_encoded = le.fit_transform(['NORTH', 'SOUTH', 'EAST', 'WEST', 'CONVERT'])
label_encoded

# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard
ship_states = {}

ship_ = 0

def decodeDir(act_):
    if act_ == 'NORTH': return directions[0]
    if act_ == 'EAST': return directions[1]
    if act_ == 'SOUTH': return directions[2]
    if act_ == 'WEST': return directions[3]

def update_L1():
    ship_ += 1

def getDirTo(fromPos, toPos, size):
    fromX, fromY = divmod(fromPos[0], size), divmod(fromPos[1], size)
    toX, toY = divmod(toPos[0], size), divmod(toPos[1], size)
    if fromY < toY: return ShipAction.NORTH
    if fromY > toY: return ShipAction.SOUTH
    if fromX < toX: return ShipAction.EAST
    if fromX > toX: return ShipAction.WEST




# Returns the commands we send to our ships and shipyards
def simple_agent(obs, config):
    size = config.size
    board = Board(obs, config)
    me = board.current_player
    # If there are no ships, use first shipyard to spawn a ship.
    if len(me.ships) == 0 and len(me.shipyards) > 0:
        me.shipyards[0].next_action = ShipyardAction.SPAWN

    # If there are no shipyards, convert first ship into shipyard.
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT

    for ship in me.ships:
        if ship.next_action == None:

            ### Part 1: Set the ship's state
            if ship.halite < 200:  # If cargo is too low, collect halite
                ship_states[ship.id] = "COLLECT"
            if ship.halite > 500:  # If cargo gets very big, deposit halite
                ship_states[ship.id] = "DEPOSIT"

            ### Part 2: Use the ship's state to select an action
            if ship_states[ship.id] == "COLLECT":
                # If halite at current location running low,
                # move to the adjacent square containing the most halite
                if ship.cell.halite < 100:
                    neighbors = [ship.cell.north.halite, ship.cell.east.halite,
                                 ship.cell.south.halite, ship.cell.west.halite]
                    best = max(range(len(neighbors)), key=neighbors.__getitem__)
                    ship.next_action = directions[best]
            if ship_states[ship.id] == "DEPOSIT":
                # Move towards shipyard to deposit cargo
                direction = getDirTo(ship.position, me.shipyards[0].position, size)
                if direction: ship.next_action = direction

    return me.next_actions



# Returns the commands we send to our ships and shipyards
def advanced_agent(obs, config, action):
    size = config.size
    board = Board(obs, config)
    me = board.current_player
    act = le.inverse_transform([action])[0]
    global ship_

    # If there are no ships, use first shipyard to spawn a ship.
    if len(me.ships) == 0 and len(me.shipyards) > 0:
        me.shipyards[ship_ - 1].next_action = ShipyardAction.SPAWN

    # If there are no shipyards, convert first ship into shipyard.
    if len(me.shipyards) == 0 and len(me.ships) > 0 and ship_ == 0:
        me.ships[0].next_action = ShipAction.CONVERT
    try:
        if act == 'CONVERT':
            me.ships[0].next_action = ShipAction.CONVERT
            update_L1()
            if len(me.ships) == 0 and len(me.shipyards) > 0:
                me.shipyards[ship_ - 1].next_action = ShipyardAction.SPAWN
        if me.ships[0].halite < 200:
            ship_states[me.ships[0].id] = 'COLLECT'
        if me.ships[0].halite > 800:
            ship_states[me.ships[0].id] = 'DEPOSIT'

        if ship_states[me.ships[0].id] == 'COLLECT':
            if me.ships[0].cell.halite < 100:
                me.ships[0].next_action = decodeDir(act)
        if ship_states[me.ships[0].id] == 'DEPOSIT':
            # Move towards shipyard to deposit cargo
            direction = getDirTo(me.ships[0].position, me.shipyards[ship_ - 1].position, size)
            if direction: me.ships[0].next_action = direction
    except:
        pass

    return me.next_actions