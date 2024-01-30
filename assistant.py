#!/usr/bin/env python3
"""Assistant implentation of the Meta-Prompt paper."""

from colorama import Fore, Style
import json
from openai import OpenAI, OpenAIError
import subprocess
import sys
from tempfile import NamedTemporaryFile
import time


continuation_prompt = '''Based on the information given, what are the most logical next steps or conclusions?
Analyze all previous interaction and ask for clarification from a different expert if one contradicts itself.
Please make sure that the solution is accurate, directly answers the original question, and follows all given constraints.
Additionally, review the final solution or have another expert(s) verify it.
If you were asked for any clarification, please provide it. Remember that experts cannot recall any previous interaction so provide all necessary details.'''


def handle_user_input():
    """Gather user input."""
    print(f"{Fore.CYAN} User Command:")
    print(Style.RESET_ALL)
    user_input = input()
    return user_input


def handle_contact_expert(client, *, name, persona, instructions):
    """Make zero-shot call to OpenAI chat completion endpoint."""
    payload = {
        "model": "gpt-4-turbo-preview",
        "messages": [
            {"role": "system", "content": f"You are an {name}. {persona}"},
            {"role": "user", "content": instructions}
        ],
        "temperature": 0.2
    }
    try:
        response = client.chat.completions.create(**payload)
        response_py = response.model_dump()
        content = response_py['choices'][0]['message']['content']
        return content
    except OpenAIError as e:
        print(str(e))


def execute_python_code_local(s: str) -> str:
    """Execute python code in a sandbox."""
    with NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(s.encode('utf-8'))
        temp_file.flush()
        try:
            result = subprocess.run(
                ['python', temp_file_name],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            return e.stderr
        finally:
            import os
            os.remove(temp_file_name)


def check_run(client, run):
    """Check the run status."""
    while True:
        run = client.beta.threads.runs.retrieve(
            thread_id=run.thread_id,
            run_id=run.id
        )

        if run.status == 'completed':
            print(f"{Fore.GREEN} Run is completed.{Style.RESET_ALL}")
            return run
        elif run.status == "expired" or run.status == "failed":
            print(f"{Fore.RED}Run is {run.status}.{Style.RESET_ALL}")
            return run
        elif run.status == "requires_action":
            print(f"{Fore.YELLOW} OpenAI: Running function...{run.status} {Style.RESET_ALL}")
            return run
        else:
            print(f"{Fore.YELLOW} OpenAI: Run is not yet completed. Waiting...{run.status} {Style.RESET_ALL}")
            time.sleep(1)  # Wait for 1 second before checking again


def run_assistant(client, run):
    """Run the assistant on the given thread."""
    last_id = None

    while True:
        run = check_run(client, run)

        if run.status == "requires_action":
            tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
            tool_outputs = []

            if tool_call.function.name == "contact_expert":
                args = json.loads(tool_call.function.arguments)
                name = args['name']
                persona = args.get('persona')
                instructions = args['instructions']

                print(f"{Fore.BLUE} {name}: {persona or ''}\n{instructions} {Style.RESET_ALL}")

                response = handle_contact_expert(client, name=name, persona=persona, instructions=instructions)

                if response:
                    print(f"{Fore.CYAN} {name}: {response} {Style.RESET_ALL}")
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_call.id,
                            "output": "\n\n".join((response, continuation_prompt))
                        }
                    )

            if tool_outputs:
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=run.thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )

        else:
            query_params = {"after": last_id} if last_id else {}
            messages = client.beta.threads.messages.list(run.thread_id, **query_params)
            messages_dict = json.loads(messages.model_dump_json())
            last_id = messages_dict['data'][-1]['id']
            for message in messages_dict['data']:
                assistant_message = message['content'][0]['text']['value']
                print(f"{Fore.BLUE} Assistant: {assistant_message} {Style.RESET_ALL}")

            if run.status == "completed":
                break


def main():
    """Initialize everything and start main loop."""
    client = OpenAI()

    user_input = handle_user_input()
    if user_input == 'quit':
        sys.exit(0)

    try:
        run = client.beta.threads.create_and_run(
            assistant_id="asst_5ulsgCs9B3ESY3tfs9Ohtvyb",
            thread={
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            }
        )
        run_assistant(client, run)

    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()
