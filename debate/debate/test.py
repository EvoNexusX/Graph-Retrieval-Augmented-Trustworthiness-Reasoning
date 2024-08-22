from Debate import Debate

arena = Debate.from_config("test.json")
arena.question = "On what date in 1969 did Neil Armstrong first set foot on the Moon?"
arena.evidences = [
            '(0) On July 20, 1969, American astronauts Neil Armstrong (1930-2012) and Edwin ¨BuzzÄldrin (1930-) became the first humans ever to land on the moon.','(1) Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin ...','(2) Neil Armstrong on the Moon. At 02:56 GMT on 21 July 1969, Armstrong became the first person to step onto the Moon. He was joined by Aldrin 19 minutes.','(3) It reads, "Here men from the planet Earth first set foot upon the moon. July 1969 A.D. We came in peace for all mankind." Armstrong and Aldrin', '(4) Apollo 11 launched from Cape Kennedy on July 16, 1969, carrying Commander Neil Armstrong, Command Module Pilot']

print(arena.players)

# Run the game for 10 steps
# arena.run(num_steps=10)

# Alternatively, you can run your own main loop
for _ in range(25):
    flag = arena.step()
    if flag.terminal is True:
        break
# arena.save_history(path=...)
# arena.save_config(path=...)

#
# # 定义问题
# question = "On what date in 1969 did Neil Armstrong first set foot on the moon?"
#
# # 设置问题
#
#
# # 执行流程
# print('===============')
#
# # 1. Simultaneous Talk Phase
# simultaneous_prompt = """
# Answer the question as accurately as possible based on the information given, and put the answer in the form [answer]. Here is an example:
# {example}
# (END OF EXAMPLE)
# [evidences]
#
# Question: {question}
# Answer: Let's think step by step!
# """
# env.set_user_input(simultaneous_prompt.format(question=question))
# print("Simultaneous Talk Phase:")
# arena.run(num_steps=1)
# arena.step()
# # 检查判官的响应
# judge_response = judge.backend.chat(env.get_history())
# if "No" in judge_response:
#     # 2. Orderly Talk Phase
#     orderly_prompt = """
# There are a few other agents assigned the same task; it's your responsibility to discuss with them and think critically. You can update your answer with other agents' answers or given evidences as advice, or you can not update your answer. Please put the answer in the form [answer].
# [evidences]
#
# [answer_from_other_agents]
# [your_historical_answer]
#
# Question: {question}
# Answer: Let's think step by step!
# """
#     env.set_user_input(orderly_prompt.format(question=question))
#     print("Orderly Talk Phase:")
#     arena.run(num_steps=2)
#     arena.step()
#
#     # 再次检查判官的响应
#     judge_response = judge.backend.chat(env.get_history())
#     if "No" in judge_response:
#         # 继续辩论
#         arena.run(num_steps=3)
#     else:
#         print("Debate over. Judge confirmed consistency.")
# else:
#     print("Debate over. Judge confirmed consistency.")
#
# # 3. Summary Phase
# summary_prompt = """
# Please summarize the final answer from answer of all agents. Place the final answer of question in the form of [answer]. Here is some examples:
# {examples}
# (END OF EXAMPLE)
#
# Question: {question}
# [all_answers_from_agents]
# Answer: Let's think step by step!
# """
# env.set_user_input(summary_prompt.format(question=question))
# print("Summary Phase:")
# summarizer_response = summarizer.backend.chat(env.get_history())
# print(summarizer_response)
#
# # 保存结果
arena.save_history(path="test.csv")
arena.save_config(path="test.text")

# # Your code goes here ...
