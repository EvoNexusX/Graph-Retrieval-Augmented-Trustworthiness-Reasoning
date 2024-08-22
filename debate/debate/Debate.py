import logging
import re
from typing import Union

from chatarena.chatarena.arena import Arena
from chatarena.chatarena.config import ArenaConfig
from chatarena.chatarena.agent import Player,SIGNAL_END_OF_CONVERSATION

from Player import Debater,Judge,Summarizer

from chatarena.chatarena.environments import Environment, TimeStep, load_environment


class TooManyInvalidActions(Exception):
    pass


class Debate(Arena):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round = 0
        self.t = 0 # handle
        self.question = None
        self.evidences = None
        self.his_ans = dict()
        self.is_terminal:bool = False
    def step(self) -> TimeStep:
        self.round = (self.t)//4
        self.t += 1
        flag = 0
        """Take a step in the game: one player takes an action and the environment updates."""
        player_name = self.environment.get_next_player()
        print(player_name,self.t,self.round)
        player = self.name_to_player[player_name]  # get the player object
        observation = self.environment.get_observation(
            player_name
        )  # get the observation for the player

        timestep = None
        for i in range(
                self.invalid_actions_retry
        ):  # try to take an action for a few times
            action = None
            if "debate" in player_name:
                action = Debater(player, observation, self.round, self.question, self.evidences,self.his_ans).act()
                self.his_ans[player_name] = action
                # action = player(observation)  # take an action
            elif self.round == 0:
                action = ""
            else:
                if "judge" in player_name:
                    action = Judge(player, observation, self.round, self.question, self.evidences,self.his_ans).act()
                    if action.endswith('<End>'):
                        self.is_terminal = True
                else:
                    if self.is_terminal:
                        action = Summarizer(player, observation, self.round, self.question, self.evidences,
                                       self.his_ans).act()
                        print(action)
                        if self.environment.check_action(action, player_name):  # action is valid
                            timestep = self.environment.step(
                                player_name, action
                            )  # update the environment

                        else:  # action is invalid
                            logging.warning(f"{player_name} made an invalid action {action}")
                            continue

                        action = SIGNAL_END_OF_CONVERSATION

                    else:
                        action = ""

            print(action)

            if self.environment.check_action(action, player_name):  # action is valid
                timestep = self.environment.step(
                    player_name, action
                )  # update the environment

                break
            else:  # action is invalid
                logging.warning(f"{player_name} made an invalid action {action}")
                continue

        if (
                timestep is None
        ):  # if the player made invalid actions for too many times, terminate the game
            warning_msg = f"{player_name} has made invalid actions for {self.invalid_actions_retry} times. Terminating the game."
            logging.warning(warning_msg)
            raise TooManyInvalidActions(warning_msg)


        return timestep


    @classmethod
    def from_config(cls, config: Union[str, ArenaConfig]):
        """Create an arena from a config."""
        # If config is a path, load the config
        if isinstance(config, str):
            config = ArenaConfig.load(config)

        global_prompt = config.get("global_prompt", None)

        # Create the players
        players = []
        for player_config in config.players:
            # Add public_prompt to the player config
            if global_prompt is not None:
                player_config["global_prompt"] = global_prompt

            player = Player.from_config(player_config)
            players.append(player)

        # Check that the player names are unique
        player_names = [player.name for player in players]
        assert len(player_names) == len(
            set(player_names)
        ), "Player names must be unique"

        # Create the environment
        config.environment[
            "player_names"
        ] = player_names  # add the player names to the environment config
        env = load_environment(config.environment)

        return cls(players, env, global_prompt=global_prompt)
