from modules.db import PostgresManager
from modules import llm
import dotenv
import os
import argparse
import asyncio

from autogen_agentchat.agents import UserProxyAgent, AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

dotenv.load_dotenv()
assert os.environ.get("DATABASE_URL"), "POSTGRES_CONNECTION_URL not found in .env file"
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not found in .env file"

DB_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

POSTGRES_TABLE_DEFINITIONS_CAP_REF = "TABLE_DEFINITIONS"
RESPONSE_FORMAT_CAP_REF = "RESPONSE_FORMAT"
SQL_DELIMITER = "---------"

async def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="The prompt for the AI")
    args = parser.parse_args()

    if not args.prompt:
        print("Please provide a prompt")
        return

    prompt = f"Fulfill this database query: {args.prompt}. "

    
    with PostgresManager() as db:
        db.connect_with_url(DB_URL)
        
        table_definitions = db.get_table_definitions_for_prompt()

        prompt = llm.add_cap_ref(
            prompt,
            f"Use these {POSTGRES_TABLE_DEFINITIONS_CAP_REF} to satisfy the database query",
            POSTGRES_TABLE_DEFINITIONS_CAP_REF,
            table_definitions,
        )

        print(prompt)

        # gpt4_config = {
        #     "use_cache": False,
        #     "temperature": 0,
        #     "config_list": autogen.config_list_from_models(["gpt-4"]),
        #     "request_timeout": 120,
        #     "functions": [
        #         {
        #             "name": "run_sql",
        #             "description": "Run a SQL query against the postgres database",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "sql": {
        #                         "type": "string",
        #                         "description": "The SQL query to run",
        #                     }
        #                 },
        #                 "required": ["sql"],
        #             },
        #         }
        #     ],
        # }

        openai_client = OpenAIChatCompletionClient(model="gpt-4", api_key=OPENAI_API_KEY, temperature=0, request_timeout=120)

        # build the function map
        function_map = {
            "run_sql": db.run_sql,
        }

        # create our terminate msg function
        termination = TextMentionTermination("APPROVED")

        COMPLETION_PROMPT = "If everything looks good, respond with the keyword 'APPROVED'."

        ADMINISTRATOR_PROMPT = (
            "Interact with the Data Engineer to discuss the plan to satisy the prompt."
        )
        DATA_ENGINEER_PROMPT = (
            "You follow an approved plan. Generate the initial SQL based on the requirements provided. Send it to the Sr Data Analyst to be executed."
        )
        SR_DATA_ANALYST_PROMPT = (
            "You follow an approved plan. You run the SQL query(if an error is encountered you correct the query and run the SQL again), generate the response and send it to the Product Manager for final review. If you encounter an error then report it back to the Data Engineer"
        )
        PRODUCT_MANAGER_PROMPT = (
            "Validate the response to make sure it's correct. If the team is unable to get the final output report it to the Sr Data Analyst and ask him to run the corrected SQL again"
            + COMPLETION_PROMPT
        )

        # create a set of agents with specific roles
        # admin user proxy agent - takes in the prompt and manages the group chat
        user_proxy = AssistantAgent(
            name="Admin",
            description = "You are the administrator",
            model_client= openai_client,
            system_message= ADMINISTRATOR_PROMPT,
        )

        # data engineer agent - generates the sql query
        data_engineer = AssistantAgent(
            name="Engineer",
            model_client=openai_client,
            description = "You are a Data Engineer",
            system_message=DATA_ENGINEER_PROMPT,
            # code_execution_config=False,
            # human_input_mode="NEVER",
        )

        # sr data analyst agent - run the sql query and generate the response
        sr_data_analyst = AssistantAgent(
            name="Sr_Data_Analyst",
            model_client=openai_client,
            description = "You are a Sr Data Analyst",
            system_message=SR_DATA_ANALYST_PROMPT,
            tools = [db.run_sql],
            reflect_on_tool_use = True,
            # code_execution_config=False,
            # human_input_mode="NEVER",
            # function_map=function_map,
        )

        # product manager - validate the response to make sure it's correct
        product_manager = AssistantAgent(
            name="Product_Manager",
            model_client=openai_client,
            description = "You are a Product Manager",
            system_message=PRODUCT_MANAGER_PROMPT,
            # code_execution_config=False,
            # human_input_mode="NEVER",
        )

        # create a group chat and initiate the chat.
        groupchat = RoundRobinGroupChat([user_proxy, data_engineer, sr_data_analyst, product_manager], termination_condition = termination, max_turns=10)
        # manager = GroupChatManager(groupchat=groupchat, llm_config=openai_client)
        team  = SelectorGroupChat(
            [user_proxy, data_engineer, sr_data_analyst, product_manager],
            model_client = openai_client,
            termination_condition = termination,
        )

        # stream = groupchat.run_stream(task=prompt)
        stream = team.run_stream(task = prompt)

        await Console(stream)


asyncio.run(main())
