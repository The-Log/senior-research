from __future__ import print_function
from vizdoom import *
from time import sleep

from agents import *

game = DoomGame()
game.set_doom_scenario_path("scenarios/basic.wad")
game.set_doom_map("MAP01")
# game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_screen_resolution(ScreenResolution.RES_1920X1080)
game.set_screen_format(ScreenFormat.RGB24)


game.set_depth_buffer_enabled(True)
game.set_labels_buffer_enabled(True)
game.set_automap_buffer_enabled(True)

# Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
game.set_render_hud(False)
game.set_render_minimal_hud(False)  # If hud is enabled
game.set_render_crosshair(True)
game.set_render_weapon(True)
game.set_render_decals(False)  # Bullet holes and blood on the walls
game.set_render_particles(False)
game.set_render_effects_sprites(False)  # Smoke and blood
game.set_render_messages(False)  # In-game messages
game.set_render_corpses(False)
game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

# Adds buttons that will be allowed.
for i in buttons:
    game.add_available_button(i)

actions = get_actions()

game.add_game_args("+freelook 1"
                   "+sv_noautoaim 0 ")

# Adds game variables that will be included in state.
game.add_available_game_variable(GameVariable.AMMO2)

game.set_episode_timeout(200)

game.set_episode_start_time(10)

game.set_window_visible(True)
# game.set_sound_enabled(True)
# Sets the livin reward (for each move) to -1
# game.set_living_reward(-1)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game.set_mode(Mode.PLAYER)

# Enables engine output to console.
#game.set_console_enabled(True)

# Initialize the game. Further configuration won't take any effect from now on.
game.init()


# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# 5 more combinations are naturally possible but only 3 are included for transparency when watching.

agent = BaseAgent(game, actions)
#agent = KeyBoardAgent(game, actions)

episodes = 1

# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.
sleep_time = 1.0 / DEFAULT_TICRATE # = 0.028

for i in range(episodes):
    print("Episode #" + str(i + 1))

    game.new_episode()

    while not game.is_episode_finished():

        # Gets the state
        state = game.get_state()

        # Which consists of:
        n = state.number
        vars = state.game_variables
        sb = state.screen_buffer
        db = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels

        # Makes a random action and get remember reward.
        r = agent.action(state)

        # Makes a "prolonged" action and skip frames:
        # skiprate = 4
        # r = game.make_action(choice(actions), skiprate)

        # The same could be achieved with:
        # game.set_action(choice(actions))
        # game.advance_action(skiprate)
        # r = game.get_last_reward()

        # Prints state's game variables and reward.
        print("State #" + str(n))
        print("Game variables:", vars)
        print("Reward:", r)
        print("=====================")

        if sleep_time > 0:
            sleep(sleep_time)

    print("Episode finished.")
    print("Total reward:", game.get_total_reward())
    print("************************")


game.close()
