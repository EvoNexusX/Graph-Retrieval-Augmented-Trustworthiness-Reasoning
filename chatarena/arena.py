from typing import List, Dict, Union
import uuid
import json
import csv
import logging
import pickle
import re

from .agent import Player
from .environments import Environment, TimeStep, load_environment
from .backends import Human
from .config import ArenaConfig
from .message import Message, MessagePool

from .graph import Graph


class TooManyInvalidActions(Exception):
    pass


class Arena:
    """
    Utility class that manages the game environment and players
    """

    def __init__(self, players: List[Player], environment: Environment, global_prompt: str = None):
        # Create a container for the players and environment and reset the game
        self.players = players
        self.environment = environment
        self.global_prompt = global_prompt

        self.current_timestep = environment.reset()
        self.uuid = uuid.uuid4()  # Generate a unique id for the game
        self.invalid_actions_retry = 5

        message = Message(agent_name="Moderator", content="Werewolf Game!", turn=-1, visible_to="None")
        self.environment.message_pool.append_message_at_index(message, 0)

        for player in players:
            player_role_desc = player.role_desc
            role_prompt = player_role_desc.split('\n')[0]
            message = Message(agent_name="Moderator", content=role_prompt, turn=-1, visible_to=player.name,
                              importance=5)
            self.environment.message_pool.append_message_at_index(message, 1)

        wolves = [self.environment.player_names[i] for i in self.environment._identity_mapping["werewolf"]]
        identity = self.environment._characters
        self.graph = Graph(self.environment.message_pool, identity, wolves, self.players, self.global_prompt)
        self.graph.infGraphs = self.graph.init_inf_graph()
        global_prompt = global_prompt.split('\n')
        global_prompt.reverse()
        for prompt in global_prompt:
            prompt = prompt.strip()
            if prompt != "" and prompt[0] != '#':
                message = Message(agent_name="Moderator", content=prompt, turn=-1, visible_to="all")
                self.environment.message_pool.append_message_at_index(message, 1)

    @property
    def num_players(self):
        return self.environment.num_players

    @property
    def name_to_player(self) -> Dict[str, Player]:
        return {player.name: player for player in self.players}

    def reset(self) -> TimeStep:
        # Reset the environment
        self.current_timestep = self.environment.reset()
        # Reset the players
        for player in self.players:
            player.reset()
        # Reset the uuid
        self.uuid = uuid.uuid4()
        return self.current_timestep

    def step(self, args=None, op=False, ) -> TimeStep:
        """
        Take a step in the game: one player takes an action and the environment updates
        """
        player_name = self.environment.get_next_player()
        player_role = self.environment._characters[self.environment.player_names.index(player_name)]
        player = self.name_to_player[player_name]  # get the player object
        observation = self.environment.get_observation(player_name)  # get the observation for the player

        turn = self.environment._current_turn
        phase = self.environment._current_phase
        alive_list = self.environment._alive_list

        # print(f"observation: {observation}")

        timestep = None
        if not self.environment._judge_is_alive(player_name):
            timestep = self.environment.step(player_name, "")
            return timestep

        state = (turn, phase, player_role, alive_list)

        for i in range(self.invalid_actions_retry):  # try to take an action for a few times
            action = player(args, observation, self.environment.message_pool, self.environment.question_pool, state,
                            self.graph)  # take an action            print(action)

            if self.environment.check_action(action, player_name):
                timestep, current_messages = self.environment.step(player_name, action)  # update the environment
                for message in current_messages:
                    flag = 0
                    if message.turn == 0:
                        continue
                    # Moderator -> Seer
                    if player.backend.type_name == "agent-chat" and message.agent_name == "Moderator" and len(
                            message.visible_to) == 1 and message.importance == 5:
                        confidence = None
                        if "is a werewolf" in message.content:
                            ans_pattern = re.compile(r"(.*?) is a werewolf", re.S)
                            p = ans_pattern.findall(message.content)[-1]
                            self.graph.infGraphs[player_name].nodes[p]['confidence'] = -1
                            self.graph.infGraphs[player_name].nodes[p]['judge'] = "enemy"
                            confidence = -1
                        if "is not a werewolf" in message.content:
                            ans_pattern = re.compile(r"(.*?) is not a werewolf", re.S)
                            p = ans_pattern.findall(message.content)[-1]
                            self.graph.infGraphs[player_name].nodes[p]['confidence'] = 1
                            self.graph.infGraphs[player_name].nodes[p]['judge'] = "teammate"
                            confidence = 1
                        lr = 0.1
                        for p in self.players:
                            p = player.name
                            try:
                                if p != player_name and self.graph.infGraphs[player_name][player_name][p]['extent'] != 0:

                                    self.graph.infGraphs[player_name][player_name][p]['extent'] = confidence / self.graph.infGraphs.nodes[p]['confidence']*lr + self.graph.infGraphs[player_name][player_name][p]['extent']
                                    if  self.graph.infGraphs[player_name][player_name][p]['extent'] > 1:
                                        self.graph.infGraphs[player_name][player_name][p]['extent'] = 1
                                    elif  self.graph.infGraphs[player_name][player_name][p]['extent'] < -1:
                                        self.graph.infGraphs[player_name][player_name][p]['extent'] = -1
                                    self.graph.graphs[player_name][player_name][p][0]['extent'] =  self.graph.infGraphs[player_name][player_name][p]['extent']
                            except:
                                    print('---')
                    elif message.agent_name=="Moderator" and phase == "daytime" and message.msg_type == "important" and "died" not in message.content:
                        pattern = re.compile(r"(.*?) are (.*?)\.", re.S)
                        anss = pattern.findall(message.content)[0]
                        pps,identities = anss
                        pps = pps.split(', ')
                        identities = identities.split(', ')
                        for i in range(len(pps)):
                            p,identity = pps[i],identities[i]
                            for one in self.players:
                                if one.backend.type_name != "agent-chat":
                                    continue
                                one = one.name

                                judge = self.graph.judge(one,identity)
                                # confidence = None
                                self.graph.infGraphs[one].nodes[p]['judge'] = judge
                                if judge == "teammate":
                                    self.graph.infGraphs[one].nodes[p]['confidence'] = 1
                                    confidence = 1
                                else:
                                    self.graph.infGraphs[one].nodes[p]['confidence'] = -1
                                    confidence = -1
                                lr = 0.1
                                for ps in self.players:
                                    ps = ps.name
                                    try:
                                        if ps != one and self.graph.infGraphs[one][one][ps]['extent'] != 0:
                                            self.graph.infGraphs[one][one][ps]['extent'] += lr* confidence / self.graph.infGraphs.nodes[p]['confidence']
                                            if  self.graph.infGraphs[one][one][ps]['extent'] > 1:
                                                self.graph.infGraphs[one][one][ps]['extent']= 1
                                            elif  self.graph.infGraphs[one][one][ps]['extent'] < -1:
                                                self.graph.infGraphs[one][one][ps]['extent'] = -1
                                            self.graph.graphs[one][one][ps][0]['extent'] = self.graph.infGraphs[one][one][ps]['extent']

                                    except:
                                        print('---')


                    elif message.msg_type == "important" or message.msg_type == "action":
                            print(message.agent_name, message.content)
                            self.graph.update(message, state)

                self.environment.message_pool.current_messages = []
                break
            else:
                self.environment.message_pool._messages.pop()
                self.environment.message_pool._messages.pop()
                logging.warning(f"{player_name} made an invalid action {action}")
                continue

        if timestep is None:  # if the player made invalid actions for too many times, terminate the game
            warning_msg = f"{player_name} has made invalid actions for {self.invalid_actions_retry} times. Terminating the game."
            logging.warning(warning_msg)
            raise TooManyInvalidActions(warning_msg)
        try:
            if timestep.terminal == True:
                with open(f"./record2/{self.args.current_game_number}.pickle", "wb") as f:
                    pickle.dump(self.graph, f)
        except:

            if timestep[0].terminal == True:
                with open(f"./record2/{self.args.current_game_number}.pickle", "wb") as f:
                    pickle.dump(self.graph, f)

        return timestep

    def next_is_human(self):
        """
        check if the next player is human
        """
        player_name = self.environment.get_next_player()
        player = self.name_to_player[player_name]
        return isinstance(player.backend, Human)

    def run(self, num_steps: int = 1):
        """
        run the game for num_turns
        """
        for i in range(num_steps):
            timestep = self.step()
            if timestep.terminal:
                break

    @classmethod
    def from_config(cls, config: Union[str, ArenaConfig], args=None):
        """
        create an arena from a config
        """
        # If config is a path, load the config
        if isinstance(config, str):
            config = ArenaConfig.load(config)

        global_prompt = config.get("global_prompt", None)
        # print(f"global_prompt: {config}")

        # Create the players
        players = []
        for player_config in config.players:
            # Add public_prompt to the player config
            if global_prompt is not None:
                player_config["global_prompt"] = global_prompt

            player = Player.from_config(player_config, args)
            players.append(player)

        # Check that the player names are unique
        player_names = [player.name for player in players]
        assert len(player_names) == len(set(player_names)), "Player names must be unique"

        # Create the environment
        config.environment["player_names"] = player_names  # add the player names to the environment config
        env = load_environment(config.environment, args)

        return cls(players, env, global_prompt=config["players"][0]["global_prompt"])

    def to_config(self) -> ArenaConfig:
        """
        convert the arena to a config
        """
        # return {
        #     "players": [player.to_config() for player in self.players],
        #     "environment": self.environment.to_config(),
        #     "global_prompt": self.global_prompt
        # }
        return ArenaConfig(
            players=[player.to_config() for player in self.players],
            environment=self.environment.to_config(),
            global_prompt=self.global_prompt
        )

    def launch_cli(self, max_steps: int = None, interactive: bool = True):
        """
        launch the command line interface
        """
        from .ui.cli import ArenaCLI
        cli = ArenaCLI(self)
        cli.launch(max_steps=max_steps, interactive=interactive)

    def save_config(self, path: str):
        """
        save the config to a file
        """
        config = self.to_config()
        config.save(path)

    def save_history(self, path: str):
        """
        save the history of the game to a file
        Supports csv and json formats.
        """
        messages = self.environment.get_observation()
        message_rows = []

        if path.endswith(".csv"):
            header = ["agent_name", "content", "turn", "timestamp", "visible_to", "msg_type"]
            for message in messages:
                message_row = [
                    message.agent_name,
                    message.content,
                    message.turn,
                    str(message.timestamp),
                    message.visible_to,
                    message.msg_type,
                ]
                message_rows.append(message_row)

            with open(path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(message_rows)
        elif path.endswith(".json"):
            for message in messages:
                message_row = {
                    "agent_name": message.agent_name,
                    "content": message.content,
                    "turn": message.turn,
                    "timestamp": str(message.timestamp),
                    "visible_to": message.visible_to,
                    "msg_type": message.msg_type,
                }
                message_rows.append(message_row)

            with open(path, "w") as f:
                json.dump(message_rows, f, indent=4)
        else:
            raise ValueError("Invalid file format")