import heapq
import itertools
import math
import re
import os
import sys
import time

import numpy as np
import networkx as nx
from typing import List, Dict, Union
from queue import Queue, PriorityQueue

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
# from ollama import

from .message import Message, MessagePool
from .agent import Player

try:
    os.environ.pop("http_proxy")
    os.environ.pop("https_proxy")
except:
    print('----')


# ===============

# ===============

class Graph:
    def __init__(self, message_pool: MessagePool, identity: dict, wolves: list, players: List[Player],
                 global_prompt: str):
        self.message_pool = message_pool
        self.wolves = wolves
        self.identity = identity
        self.players = players
        self.global_prompt = global_prompt
        self.graphs = self.init_graph()
        self.infGraphs = self.init_inf_graph()
        self.evidences = dict()
        self.rho = 0.95
        self.beta = 0.8
        self.inference_mode = False

    @retry(stop=stop_after_attempt(10), wait=wait_random_exponential(min=60, max=120))
    def _get_response(self, messages, conn_method=0, max_tokens=None, T=None, stream=False):
        max_toks = 200 if max_tokens is None else max_tokens
        temperature = 0.2 if T is None else T
        model = "gpt-3.5-turbo"
        methods = {
            0: "openai",
            1: "llama"
        }
        method = methods[conn_method]
        # method = "temp"
        if method == "openai":
            # base_url = "https://api.xiaoai.plus/v1"
            base_url = "https://xa.blackbox.red/v1"
            api_key = "sk-lfHDOEV9YpiZsS5SDc2fA0475f754eC0A6Ff72EdAc7f0aB5"

        elif method == "llama":
            base_url = "http://localhost:11434/v1/"
            api_key = "ollama"
        elif method == "temp":
            api_key = "sk-TwdmLnOPrEuQVZa4Ea108fD69f1b4083Ae9410E0E559241f"
            base_url = "https://new.apiapp.one/v1"
        else:
            # Insert your base_url and api_key here.
            base_url = None
            api_key = None

        for message in messages:
            if "<EOS>" in message['content']:
                message['content'] = message['content'].strip('<EOS>')

        client = OpenAI(base_url=base_url,
                        api_key=api_key,
                        )
        start_time = time.time()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_toks,
            # stop=STOP,
            timeout=30,
            stream=stream
        )
        return completion

    def init_graph(self, type=0):
        '''
        node:the player
        edge:the relevance and the evidence
        '''

        graphs = dict()
        for p in self.players:
            if p.backend.type_name != "agent-chat" and p.backend.type_name != "agent-chat4" and p.backend.type_name != "agent-chat5" and p.backend.type_name != "agent-chat6":
                continue
            try:
                type = int(p.backend.type_name.split('chat')[1])
            except:
                type = 0
            graph = nx.MultiDiGraph()
            if p.name in self.wolves:

                if type == 0:
                    pos = 1
                    neg = -0.9
                elif type == 4:
                    pos = 0.9
                    neg = 0.9
                elif type == 5:
                    pos = 0
                    neg = 0
                elif type == 6:
                    pos = -0.9
                    neg = -0.9
                else:
                    print('wrong type!')
                if pos > 0:
                    j1 = "teammate"
                elif pos == 0:
                    j1 = "unsure"
                else:
                    j1 = "enemy"
                if neg > 0:
                    j2 = "teammate"
                elif neg == 0:
                    j2 = "unsure"
                else:
                    j2 = "enemy"
                for player in self.players:
                    if player.name in self.wolves:
                        graph.add_node(player.name, confidence=pos, judge=j1)
                    else:
                        graph.add_node(player.name, confidence=neg, judge=j2)

            else:
                for player in self.players:
                    graph.add_node(player.name, confidence=0, judge="unsure")

            for player1, player2 in itertools.combinations(self.players, 2):
                if p.name in self.wolves and player1.name in self.wolves and player2.name in self.wolves:
                    intend = "Defend"
                    extent = 1
                else:
                    intend = "Neutral"
                    extent = 0

                graph.add_edge(player1.name, player2.name, key=0, turn=0, phase="daytime", extent=extent,
                               evidence="",
                               analysis="")
                graph.add_edge(player2.name, player1.name, key=0, turn=0, phase="daytime", extent=extent,
                               evidence="",
                               analysis="")

            graphs[p.name] = graph
        return graphs

    def init_inf_graph(self, type=0):
        graphs = dict()
        for p in self.players:
            if p.backend.type_name != "agent-chat" and p.backend.type_name != "agent-chat4" and p.backend.type_name != "agent-chat5" and p.backend.type_name != "agent-chat6":
                continue
            try:
                type = int(p.backend.type_name.split('chat')[1])
            except:
                type = 0
            graph = nx.DiGraph()
            if p.name in self.wolves:
                if type == 0:
                    pos = 1
                    neg = -0.9
                elif type == 4:
                    pos = 0.9
                    neg = 0.9
                elif type == 5:
                    pos = 0
                    neg = 0
                elif type == 6:
                    pos = -0.9
                    neg = -0.9
                else:
                    print('wrong type!')
                if pos > 0:
                    j1 = "teammate"
                elif pos == 0:
                    j1 = "unsure"
                else:
                    j1 = "enemy"
                if neg > 0:
                    j2 = "teammate"
                elif neg == 0:
                    j2 = "unsure"
                else:
                    j2 = "enemy"
                for player in self.players:
                    if player.name in self.wolves:
                        graph.add_node(player.name, confidence=pos, judge=j1)
                    else:
                        graph.add_node(player.name, confidence=neg, judge=j2)

            else:
                for player in self.players:
                    if p.name == player.name:
                        graph.add_node(player.name, confidence=1,
                                       judge="teammate")
                    else:
                        graph.add_node(player.name, confidence=0, judge="unsure")

            graphs[p.name] = graph
        return graphs

    def update(self, message: Message, state: tuple):
        player_name = message.agent_name
        turn, phase, role, alive_list = state
        edges = self._update_edges(player_name, message)

        for p in self.players:
            if p.backend.type_name == "agent-chat" and (p.name in message.visible_to or message.visible_to == "all"):
                self._update_nodes(p, player_name, message, state)
                if edges is not None:
                    for edge in edges:
                        try:
                            player1, player2, extent, analysis = edge
                            key = self.graphs[p.name].number_of_edges(player1, player2)
                            self.graphs[p.name].add_edge(player1, player2, key=key, turn=turn, phase=phase,
                                                         extent=extent, evidence=[message.content],
                                                         analysis=analysis)
                        except:
                            continue

    def _update_nodes(self, p: Player, player_name: str, message: Message, state: tuple):
        name = p.name

        G = self.infGraphs[name]
        (turn, phase, player_role, alive_list) = state

        system_prompt_str = f"{self.global_prompt.strip()}\n"
        system_prompt = {"role": "system", "content": system_prompt_str}
        statement = f"({player_name}): {message.content})"

        role_desc = p.role_desc
        role_desc = {"role": "system", "content": role_desc}
        known = f"You are the {player_role}."
        # if player_role == "werewolf":
        #     known += f"Your werewolf teammates are {self.wolves}."
        #
        if player_name == "Moderator":
            inference = f"The moderator's statement is absolutely correct."
        elif G.nodes[player_name]['confidence'] > 0:
            inference = f"{player_name} is inferred to be my {G.nodes[player_name]['judge']}. My level of trust in him is {round(G.nodes[player_name]['confidence'], 3)}"

        elif G.nodes[player_name]['confidence'] < 0:
            inference = f"{player_name} is inferred to be my {G.nodes[player_name]['judge']}. His level of threat to me is {round(G.nodes[player_name]['confidence'], 3)}"
        else:
            inference = ""
        identities = {"role": "system", "content": known}
        request_prompt = {"role": "system",
                          "content": f"\nThere were nine players in the game: three werewolves, a witch, a guard, a seer, and three villagers. "
                                     f"Given the one player's statement. Through the statement, think step by step, you need to infer the identity of some players with a confidence level. Make sure the inference is reasonable. If you are the Seer, then inferring that player's identity cannot be the Seer, because there is only one Seer on the field. If you are a Witch, then it is impossible to infer that player's identity as a Witch, since there is only one Witch on the field. If you are a Guard, then it is inferred that the player's identity cannot be a Guard, since there is only one Guard on the field."
                                     f"Note that you need to determine if the statement is spurious based on the players's identities you know and the identity of the speaking player. Of course, the moderator's statement must be correct. For example, if you are witch or a werewolf and the player speaks claiming to be a witch, you can infer that he might be a witch. However if you are a witch and the player speaks claiming to be a witch, you can infer that he is lying and thus infer that he is probably a werewolf."
                                     f"(Infer and choose from the option:[werewolf,not-werewolf,villager, witch, seer,guard]. The confidence level is greater than or equal to 0 and less than 10, the more certain and more evidence there is about the player's identity, the higher the confidence level, and vice versa the lower the confidence level. When the confidence level is lower than 5, it is a bold inference, and conversely it is a more confident inference.). Please return this strictly in the format [Player][identity][confidence][analysis] with no additional output. If still unable to determine the identity, still view it as unsure.\n"
                                     f"Please note that the statement might address multiple players simultaneously. In such cases, list each relevant result separately instead of in one line!!!.\n\nHere are some examples:\n"
                                     f"Examples:\n1.Statement: (Moderator):(1-th daytime) Player 2, Player 5 are villager, witch. Answer:[Player 2][villager][10][moderator's statement is right.]\n[Player 5][witch][10][moderator's statement is right.]"
                                     f"2.Statement: (Player 3):(1-th daytime) I'm the seer. Last night I checked out Player 4. He's a werewolf. Answer:[Player 3][seer][8][I'm not a seer, and no one before Player 3 has declared him to be a seer, so it can be inferred that it might be a seer with a confidence level of 8.]\n[Player 4][werewolf][8][Player 3 is inferred to be the seer, and he checked Player 4 as a werewolf last night, so it can be inferred that Player 4 is a werewolf.]"
                                     f"{inference}. Now given the statement:{statement}."
                          }

        retry = 5
        for _ in range(retry):
            try:

                completion = self._get_response([role_desc, identities, request_prompt])
                results = completion.dict()['choices'][0]['message']['content'].split('\n')
                for res in results:
                    # 定义匹配的正则表达式模式
                    pattern = r"\[(.*?)\]\[(.*?)\]\[(.*?)\]\[(.*?)\]"
                    # 使用 re 模块进行匹配
                    match = re.findall(pattern, res)[0]
                    speaker, identity, confidence, analysis = match

                    # confidence = float(confidence) / 10 * G.nodes[speaker]['confidence']
                    confidence = float(confidence) / 10

                    if abs(confidence) < abs(G.nodes[speaker]['confidence']) or abs(confidence) < 0.5:
                        continue
                    judge = self.judge(name, identity)

                    if identity not in ("werewolf", "villager", "not-werewolf"):
                        confidence = min(1.0, confidence * 1.2)
                    else:
                        confidence = confidence

                    if judge == "enemy":
                        confidence = - confidence

                    if G.nodes[speaker]['confidence'] != 1.0 and G.nodes[speaker]['confidence'] != -1.0:
                        G.nodes[speaker]['confidence'] = confidence
                        G.nodes[speaker]['judge'] = judge

                    if speaker in self.evidences.keys():
                        if (statement, analysis) not in self.evidences[speaker]:
                            self.evidences[speaker].append((statement, analysis))
                    else:
                        self.evidences[speaker] = [(statement, analysis)]
                    lr = 0.1
                    for player in self.players:
                        player = player.name
                        if player != speaker and G[speaker][player]['extent'] != 0:
                            G[speaker][player]['extent'] = confidence / G.nodes[player]['confidence'] * lr + \
                                                           G[speaker][player]['extent']
                            if G[speaker][player]['extent'] > 1:
                                G[speaker][player]['extent'] = 1
                            elif G[speaker][player]['extent'] < -1:
                                G[speaker][player]['extent'] = -1
                            self.graphs[name][speaker][player][0]['extent'] = G[speaker][player]['extent']

                break
            except:

                continue
        self.infGraphs[name] = G

    def _update_edges(self, player_name: str, message: Message):

        system_prompt_str = f"{self.global_prompt.strip()}\n"
        system_prompt = {"role": "system", "content": system_prompt_str}
        statement = f"({player_name}): {message.content})"
        request_prompt = {"role": "system",
                          "content": f"Werewolf is a role-playing deduction game where players use their statements to attack, defend, or deceive other players."
                                     f"You are an expert in analyzing language within the context of the game Werewolf."
                                     f"Your task is to analyze a given player's statement and determine its type. "
                                     f"Based on the statement provided, determine which of the following types it belongs to:\n"
                                     f"Attack: The player attempts to question or accuse another character, suggesting they might be suspicious, or provide evidence against another character, suggesting they might be a werewolf..\n"
                                     f"Defend: The player tries to defend a character, suggesting they are not suspicious. Note that character A and character B must are the members of [Player 1, Player 2,...] instead of their game role, and might be the same, meaning the statement might be self-defense.\n"
                                     f"Deceive: The player attempts to mislead other players with false information.\n"
                                     f"Additionally, provide a score indicating the strength or certainty of the statement's intent on a scale of 0 to 10 where 0 is very weak/uncertain and 10 is very strong/certain.\n"
                                     f"Carefully read the following statement and determine its type based on its content and tone:[Player's statement],Please choose the appropriate type and briefly explain your reasoning in the following format:[Role 1][Type][Role 2][Reason][Score]. Please note that the statement might address multiple players simultaneously. In such cases, list each relevant result separately instead of in one line!!!.\n\nHere are some examples:\n"
                                     f"Example:\n1.Statement: [(Player 1): I think Player 2's behavior was very strange. He kept avoiding important discussions. I believe Player 4 is innocent because he has been helping us.]\nAnswer: [Player 1][Attack][Player 2][The Player 1 is questioning Player 2's behavior, implying they might be suspicious.][6]\n[Player 1][Defend][Player 2][The Player 1 is defending Player 4, suggesting they are not suspicious.][7]\n"
                                     f"2.Statement: [(Player 4):I observed that Player 3 was identified as a werewolf by the moderator. I believe we should carefully consider the roles of the remaining players and gather more information before making any decisions.]\nAnswer: [Player 4][Attack][Player 3][The current player indirectly accuses Player 4 of being a werewolf by mentioning the moderator's identification, influencing others' perceptions.][9]\n"
                                     f"3.Statement: [(Player 7):I believe Player 4 is innocent. He has been helping us analyze the situation.]\nAnswer: [Player 7][Defind][Player 4][The Player 7 is defending Player 4, suggesting they are not suspicious.][7]\n"
                                     f"4.Statement: [(Player 1):I choose to eliminate Player 3.]\nAnswer:[Player 1][Attack][Player 3][The Player 1 is strongly attacting Player 7.][10]"
                                     f"5.Statement: [(Player 2):I choose to protect Player 3.]\nAnswer:[Player 2][Defend][Player 3][The Player 1 is strongly protecting Player 7.][10]"
                                     f"(End of Example)\n\nNow given the statement:\nStatement: {statement}"}

        retry = 5
        for _ in range(retry):
            request = [request_prompt]
            completion = self._get_response(request)
            try:
                edges = []
                results = completion.dict()['choices'][0]['message']['content'].split('\n')
                for res in results:
                    # 定义匹配的正则表达式模式
                    pattern = r"\[(.*?)\]\[(.*?)\]\[(.*?)\]\[(.*?)\]\[(.*?)\]"
                    # 使用 re 模块进行匹配
                    matches = re.findall(pattern, res)
                    # matches 是一个列表，每个元素是一个元组，包含匹配到的组
                    for match in matches:
                        Player_a = match[0]
                        attack_defend = match[1]
                        Player_b = match[2]
                        reason = match[3]
                        score = float(match[4]) if attack_defend == 'Defend' else -float(match[4])
                        score = score / 10
                        # key = graph.number_of_edges(Player_a, Player_b)
                        if player_name in self.evidences.keys():
                            if (statement, reason) not in self.evidences[player_name]:
                                self.evidences[player_name].append((statement, reason))
                        else:
                            self.evidences[player_name] = [(statement, reason)]

                        edges.append((Player_a, Player_b, round(score, 3), reason))
                        # graph.add_edge(Player_a, Player_b, key=key, turn=turn, phase=phase,
                        #                intend=attack_defend, extent=score, evidence=[message], analysis=reason)

                return edges
            except:
                continue

            # print(completion.dict()['choices'][0]['message']['content'])

    def inference(self, name: str, output: str = "str", topk=3):
        G = self.graphs[name]
        for player1, player2 in itertools.combinations(self.players, 2):
            edges = G.get_edge_data(player1.name, player2.name)
            self.merge(name, edges, player1.name, player2.name)
            self.merge(name, edges, player2.name, player1.name)

        G = self.infGraphs[name]
        # find k best evidences
        nodes = [node for node in G.nodes(data=True)]
        nodes.sort(key=lambda x: abs(x[1]['confidence']), reverse=True)
        select = nodes[:topk]
        tree = PriorityQueue()
        for node in select:
            tree.put((abs(node[1]['confidence']), (time.time(), node)))
        visited = set()
        visited_edges = set()
        confidences = dict()
        while not tree.empty():
            c, (_, node) = tree.get()
            c = node[1]['confidence']
            if c == 0:
                break
            for player in self.players:
                player = player.name
                if player == node[0]:
                    continue
                if (player, node[0], node[1]['confidence']) in visited_edges:
                    continue
                extent = G[player][node[0]]['extent']
                if extent != 0:
                    np1 = round(abs(c * extent), 3)
                    if c * extent < 0:
                        np1 = -np1

                    if np1 < 0:
                        # i1 = self.get_enemy(i2)
                        new_node = (player, {"confidence": np1,
                                             "judge": "enemy"})
                    else:

                        new_node = (player, {"confidence": np1,
                                             "judge": "teammate"})
                        tree.put((abs(np1), (time.time(), new_node)))

                    if player not in visited:
                        confidences[player] = [(new_node)]
                    else:
                        confidences[player].append(new_node)
                    visited.add(player)
                    visited_edges.add((player, node[0], node[1]['confidence']))

        for p_name, c in confidences.items():
            count = 0
            total = 0
            flag = 0
            if len(c) > 1:
                flag = 1
            for chain in c:
                n, node = chain
                confidence = node['confidence']
                abs_conf = abs(confidence)
                if abs_conf != 0:
                    h1 = -abs_conf * math.log(abs_conf)
                    v1 = (abs_conf - h1)
                    count += v1 * confidence
                    total += v1

            nc = round(count / total, 3)
            if nc >= 0:
                p = p_name
                # if G.nodes[p]['identity'] in ("werewolf","villager","not-wolf"):
                # G.nodes[p]['identity'] = self.get_teammate(G.nodes[name]['identity'])
                oc = G.nodes[p]['confidence']
                if oc * nc > 0 or abs(nc) > abs(oc):
                    G.nodes[p]['confidence'] = nc
                    G.nodes[p]['judge'] = "teammate"


            else:
                p = p_name
                oc = G.nodes[p]['confidence']
                # if G.nodes[p]['identity'] in ("werewolf", "villager", "not-wolf"):
                # G.nodes[p]['identity'] = self.get_enemy(G.nodes[name]['identity'])
                if oc * nc > 0 or abs(nc) > abs(oc):
                    G.nodes[p]['confidence'] = nc
                    G.nodes[p]['judge'] = "enemy"

            if flag:
                lr = 0.1
                for player in self.players:
                    player = player.name
                    if player != p_name and G[p_name][player]['extent'] != 0:
                        try:

                            G[p_name][player]['extent'] = nc / G.nodes[player]['confidence'] * lr + G[p_name][player][
                                'extent']
                            if G[p_name][player]['extent'] > 1:
                                G[p_name][player]['extent'] = 1
                            elif G[p_name][player]['extent'] < -1:
                                G[p_name][player]['extent'] = -1
                            self.graphs[name][p_name][player][0]['extent'] = G[p_name][player]['extent']
                        except:
                            print('---')

        self.infGraphs[name] = G

        total = []
        trust = []
        distrust = []
        s = [f"I am {name}, the {self.Identity(name)}. I trust myself very much."]

        nodes = [x for x in self.infGraphs[name].nodes(data=True)]
        nodes.sort(key=lambda x: abs(x[1]['confidence']), reverse=True)

        for idx in range(len(nodes)):
            node = nodes[idx]
            if node[0] == name:
                continue

            confidence, judge = node[1]['confidence'], node[1]['judge']
            epsilon = 0.2

            if abs(confidence) <= epsilon:
                rest = [x[0] for x in nodes[idx:]]
                s.append(f"\n\n{rest}'s identities are still unsure, I cannot say any information about them.")
                break
            else:
                if judge == "teammate":
                    ii = self.get_teammate(self.Identity(name))
                else:
                    ii = self.get_enemy(self.Identity(name))
                if confidence > 0:
                    s.append(
                        f"{node[0]} is inferred to be a {ii}, my {judge}. My level of trust in him is {round(float(confidence), 3)}")
                else:
                    s.append(
                        f"{node[0]} is inferred to be a {ii}, my {judge}. His level of threat to me is {round(float(-confidence), 3)}.")
            total.append((node[0], confidence))

        total.sort(reverse=True, key=lambda x: abs(x[1]))
        trust.extend([x[0] for x in total[:3] if x[1] > 0])
        distrust.extend([x[0] for x in total[-3:] if x[1] < 0])
        s = '\n'.join(s)
        infer = f"\nThe below are my inference of the other players:\n{s}\nThe greater the level of trust, the more I should protect him, and the greater the level of threat, the more I should eliminate him."
        add1 = f"\n\nIn these players, {trust if trust != [] else None} are more trustworthy to me. {distrust if distrust != [] else None} are more threatening to me, and I need to be very aware of them."
        return infer + add1

    def get_enemy(self, identity):
        if identity == "werewolf":
            return "good people"
        else:
            return "werewolf"

    def get_teammate(self, identity):
        if identity == "werewolf":
            return "werewolf"
        else:
            return "good people"

    def Identity(self, name: str):
        role = self.identity[int(name.split()[1]) - 1]
        return role

    def get_identity(self, identity):
        if identity == "pretty girl":
            identity = "werewolf"
        elif identity == "prophet":
            identity = "seer"
        elif identity == "pharmacist":
            identity = "witch"
        return identity

    def judge(self, player1, identity):
        i1 = int(player1.split()[1])
        # i2 = int(player2.split()[1])
        r1 = self.identity[i1 - 1]
        # r2 = self.identity[i2 - 1]
        r2 = identity
        if r1 != r2 and "werewolf" in (r1, r2):
            return "enemy"
        else:
            return "teammate"

    def get_evidences(self, G, path):
        details = []
        n = len(path) - 1
        for i in range(n):
            node = path[i]
            edge = (path[i], path[i + 1])
            identity, confidence = G.nodes[node]['identity'], G.nodes[node]['confidence'],
            intend, extent, evidence = G.edges[edge].values()
            details.append(f"{node} is inferred to be {identity} with the confidence level of {confidence}.")
            if evidence != "":
                details.append(f"{evidence} This represent the intend '{intend}' of the level {abs(extent)}")
        identity, confidence = G.nodes[path[n]]['identity'], G.nodes[path[n]]['confidence'],
        details.append(f"{path[n]} is inferred to be {identity} with the confidence level of {confidence}.")

        return details

    def merge(self, name, edges, p1, p2):
        extent_o, evidence = self.calc(edges, p1)
        self.graphs[name].remove_edges_from(
            [(u, v, k) for u, v, k in self.graphs[name].edges(keys=True) if u == p1 and v == p2])
        self.graphs[name].add_edge(p1, p2, key=0, turn=None, phase=None, extent=extent_o, evidence=evidence,
                                   analysis="")

        self.infGraphs[name].add_edge(p1, p2, extent=extent_o, evidence=evidence)

    def calc(self, edges, speaker):
        rho = self.rho
        count = 0
        evidences = ""
        edges = list(edges.items())
        edges.sort(key=lambda x: x[0])

        n = len(edges)
        maxn = 0
        if len(edges) == 1:
            key, edge = edges[0]
            turn, phase, extent, evidence, analysis = edge.values()
            return extent, f"{speaker}:{evidence}\n"
        for key, edge in edges:
            turn, phase, extent, evidence, analysis = edge.values()

            count += rho ** (n - key - 1) * extent
            if evidence != "":
                evidences += f"{speaker}:{evidence}\n"

        return self.tanh(count), evidences

        # 定义标准的Sigmoid函数

    def tanh(self, x: float):
        return round((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), 3)
