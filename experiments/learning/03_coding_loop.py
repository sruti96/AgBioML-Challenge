import asyncio

from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import CancellationToken
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console


async def main():
    code_executor = DockerCommandLineCodeExecutor(
        image='my-multi-env-image:latest',
        work_dir='./tmp',
        timeout=600
    )
    await code_executor.start()

    model_client = OpenAIChatCompletionClient(
        model='gpt-4o',
    )

    code_executor_agent = CodeExecutorAgent('code_executor', code_executor=code_executor)
    code_writer_agent = AssistantAgent(
        model_client=model_client,
        name='code_writer',
        system_message="""
You are a code writer. You are given a task and you need to write the code to complete the task.
The code will be executed by the code executor agent. You will be given the result of the code execution.
You will need to review the result and write the code again to fix the errors if there are any.
You will continue to do this until the code works and produces the correct output.

Once you are satisfied with the code, you will say "DONE".

You have been provided with a docker container that has conda installed. There are already the following environments installed:
- sklearn-env
- torch-env
- tensorflow-env
- scanpy-env

Use this template to provide code to the executor agent:

```bash
#!/bin/bash

# Ensure conda is set up in the shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate <environment-name>

# Run the Python code
PYTHONFAULTHANDLER=1 python - <<END
<your_python_code>
END
```

A couple additional notes:
- We are using a newer version of keras so you import it as `keras` instead of `tensorflow.keras`.

"""
    )


    team = RoundRobinGroupChat(
        participants=[code_writer_agent, code_executor_agent],
        termination_condition=TextMentionTermination('DONE')
    )

    result = await Console(team.run_stream(task="""
Write a script that predicts handwritten digits using the MNIST dataset.
"""))

    print(result)


if __name__ == '__main__':
    asyncio.run(main())