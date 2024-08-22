from typing import List

from chatarena.chatarena.agent import Player
from chatarena.chatarena.backends import OpenAIChat
from chatarena.chatarena.message import Message

import os
import re

os.environ['OPENAI_API_KEY'] = "sk-lfHDOEV9YpiZsS5SDc2fA0475f754eC0A6Ff72EdAc7f0aB5"
os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"
from openai import OpenAI
client = OpenAI(
    # 下面两个参数的默认值来自环境变量，可以不加
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)
END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI
STOP = ("<|endoftext|>", END_OF_MESSAGE)  # End of sentence token
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256

class Debater:
    def __init__(self, player, observation, round, question, evidences, his_ans, *args, **kwargs):

        self.player = player
        self.observation = observation
        self.round = round
        self.question = question
        self.answer = None
        self.evidences = evidences
        self.selection = None
        self.his_ans:dict = his_ans

    def act(self) -> str:
        if self.round == 0:
            # Self-Selection
            evidence = '\n'.join(self.evidences)
            prompt = f'Please select evidence from the evidence pool that will help you answer the question. If the evidence pool does not contain the information needed to answer the question, add [No Found] at the end of your response. If the evidence pool has evidence that can help you answer the question, please return up to 3 of the most helpful evidence. Put the number in square brackets.\n\n  Evidence: {evidence} \n\n Question: {self.question}'
            selection = self.player.backend.query(
                agent_name=self.player.name,
                role_desc=self.player.role_desc,
                history_messages=self.observation,
                global_prompt=self.player.global_prompt,
                request_msg=prompt,
            )
            self.selection = selection

            # First Round Debate
            prompt = f'Answer the question as accurately as possible based on the information given, and put the answer in the form [answer]. Here is an example: \nQuestion: who was the first person killed in a car accident?\n Answer: Let’s think step by step! This tragic event occurred on August 17, 1896, in London, England. Bridget Driscoll was struck and killed by an automobile driven by Arthur Edsall, who was driving at a speed of approximately 4 miles per hour (about 6.4 kilometers per hour) at the Crystal Palace in London. Therefore, the answer is [Bridget Driscoll]. (END OF EXAMPLE) \n\nEvidence: {self.selection}\n\nQuestion: {self.question}\n\n Answer: Let’s think step by step!'
            ans = self.player.backend.query(
                agent_name=self.player.name,
                role_desc=self.player.role_desc,
                history_messages=self.observation,
                global_prompt=self.player.global_prompt,
                request_msg=prompt,
            )
            self.answer = ans
            return ans
        else:
            # Self-Selection
            evidence = '\n'.join(self.evidences)
            prompt = f'Please select evidence from the evidence pool that will help you answer the question. If the evidence pool does not contain the information needed to answer the question, add [No Found] at the end of your response. If the evidence pool has evidence that can help you answer the question, please return up to 3 of the most helpful evidence. Put the number in square brackets.\n\n  Evidence: {evidence} \n\n Question: {self.question}'
            selection = self.player.backend.query(
                agent_name=self.player.name,
                role_desc=self.player.role_desc,
                history_messages=self.observation,
                global_prompt=self.player.global_prompt,
                request_msg=prompt,
            )
            self.selection = selection
            other_ans = []
            for player,ans in self.his_ans.items():
                if player != self.player.name:
                    other_ans.append(f"({player}){self.his_ans[player]}")
            other = '\n'.join(other_ans)
            evidence = self.selection
            prompt = f"There are a few other agents assigned the same task, it's your responsibility to discuss with them and think critically. You can update your answer with other agents’ answers or given evidences as advice, or you can not update your answer. Please put the answer in the form [answer].\n\n Evidence:\n {evidence}\n\nAnswers from other Agents:{other} \n\nHere is your historical answer:{self.answer} Question: {self.question} Answer: Let’s think step by step!"
            ans = self.player.backend.query(
                agent_name=self.player.name,
                role_desc=self.player.role_desc,
                history_messages=self.observation,
                global_prompt=self.player.global_prompt,
                request_msg=prompt,
            )
            self.answer = ans
            return ans

    # def query(self,prompt):
    #     completion = client.chat.completions.create(
    #         model=DEFAULT_MODEL,
    #         messages= {"role":"user","content":prompt}
    #         temperature=DEFAULT_TEMPERATURE,
    #         max_tokens=DEFAULT_MAX_TOKENS,
    #
    #     )
    #
    #     response = completion.choices[0].message.content
    #     response = response.strip()
    #     return response


class Judge:
    def __init__(self, player, observation, round, question, evidences,his_ans,*args, **kwargs):

        self.player = player
        self.observation = observation
        self.round = round
        self.question = question
        self.evidences = evidences
        self.selection = None
        self.his_ans:dict = his_ans
    def act(self) -> str:

        res = []
        for player, ans in self.his_ans.items():
                res.append(f"({player}):{self.his_ans[player]}")
        results = '\n'.join(res)
        prompt = f'The answer of the agents are typically denoted with the [answer] format. Your task is to extract each agent’s answer and evaluate the consistency of their answers to the question. If all agents have provided correct and consistent answers, respond with [Yes]. If their answers are inconsistent, respond with [No]. Please ensure to encase your response - Yes or No - within square brackets. Question: {self.question}\n\n Agent Responses:{results}\n\nAnswer: Let’s think step by step!'
        ans = self.player.backend.query(
            agent_name=self.player.name,
            role_desc=self.player.role_desc,
            history_messages=self.observation,
            global_prompt=self.player.global_prompt,
            request_msg=prompt,
        )
        if self.is_terminal(ans):
            ans = ans + '<End>'
        return ans

    def is_terminal(self,ans) -> bool:

        if re.search(
                r"yes|Yes|y|yea|yeah|yep|yup|sure|ok|okay|alright", ans, re.IGNORECASE
        ):
            return True
        else:
            return False


class Summarizer:
    def __init__(self, player, observation, round, question, evidences,his_ans,*args, **kwargs):

        self.player = player
        self.observation = observation
        self.round = round
        self.question = question
        self.evidences = evidences
        self.selection = None
        self.his_ans:dict = his_ans
    def act(self) -> str:

        res = []
        for player, ans in self.his_ans.items():
                res.append(f"({player}):{self.his_ans[player]}")
        results = '\n'.join(res)
        prompt = f'Please summarize the final answer from answer of all agents. Place the final answer of question in the form of [answer]. Here is an example: Question: How many husbands did the Wife of Bath have, as reported in Chaucer’s Canterbury Tales? Agent 0: In Chaucer’s Canterbury Tales, the Wife of Bath claims to have had [five] husbands. Agent 1: In Chaucer’s Canterbury Tales, the Wife of Bath, one of the most memorable characters, claims to have had [five] husbands. Answer: Let’s think step by step! Based on the answers provided by Agent 0 and Agent 1, it can be concluded that in Chaucer’s Canterbury Tales, the Wife of Bath claims to have had five husbands. Therefore, the final answer is [five]. Question: Ezzard Charles was a world champion in which sport? Agent 0: Ezzard Charles was a world champion in the sport of boxing. He held the world heavyweight title from 1949 to 1951. Therefore, the answer is "boxing". Agent 1: Ezzard Charles was a world champion in boxing. Therefore, the answer is "boxing". Answer: Let’s think step by step! Based on the responses from Agent 0 and Agent 1, it is clear that Ezzard Charles was a world champion in the sport of boxing. He held the world heavyweight title from 1949 to 1951. Therefore, the final answer is [boxing]. Question: In which city were Rotary Clubs set up in 1905? Agent 0: The first Rotary Club was established in Chicago, Illinois, United States in 1905. Therefore, the answer is Chicago. Agent 1: The Rotary Clubs were set up in 1905 in the city of Chicago, Illinois, United States. Therefore, the answer is City of Chicago. Answer: Let’s think step by step! Based on the responses from both Agent 0 and Agent 1, it is clear that the Rotary Clubs were first established in the city of Chicago, Illinois, United States in the year 1905. Therefore, the final answer is [Chicago]. (END OF EXAMPLE) \n\nQuestion: {self.question}\n\n Agent Responses: {results}. \n\nAnswer: Let’s think step by step!'
        ans = self.player.backend.query(
            agent_name=self.player.name,
            role_desc=self.player.role_desc,
            history_messages=self.observation,
            global_prompt=self.player.global_prompt,
            request_msg=prompt,
        )
        return ans