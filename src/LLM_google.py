import os
import re
from dotenv import load_dotenv
import logging
import torch
from llama_cpp import Llama
from memory import Memory
from datetime import datetime, timedelta
from personality_utils import load_embedded_personality
from hyperdb import HyperDB
from google_calendar import GoogleCalendarManager
from dateutil import parser
from dateutil.relativedelta import relativedelta

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMWrapper:
    def __init__(self, model_path):
        self.model_path = model_path
        self.llm = None
        self.memory = Memory()
        self.name = os.getenv('NAME_OF_BOT')
        self.role = os.getenv('ROLE_OF_BOT')
        self.personality, self.personality_embedding = load_embedded_personality()
        self.db = HyperDB()
        self.gcal = GoogleCalendarManager()

    def initialize(self):
        if self.llm is None:
            logger.info("Initializing LLM...")
            gpu_layers = -1 if torch.cuda.is_available() else 0 
            logger.info(f"Using GPU layers: {gpu_layers}")
            self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_batch=512, n_gpu_layers=gpu_layers, verbose=True)
            logger.info("LLM initialized successfully.")

    def ask(self, prompts, format="", temperature=0.7):
        self.initialize()
        prompt = " ".join([p["content"] for p in prompts])
        logger.debug(f"Sending prompt to Llama: {prompt}")
        logger.debug(f"Temperature: {temperature}")
        output = self.llm(prompt, max_tokens=2048, temperature=temperature, top_p=0.9, echo=False)
        logger.debug(f"Raw output from Llama: {output}")
        response = output['choices'][0]['text'].strip()
        logger.debug(f"Stripped response: {response}")
        return response
    
    def close(self):
        if self.llm is not None:
            del self.llm
            self.llm = None
            logger.info("LLM resources released.")

    def classify_query(self, query):
        classification_prompt = f"""
        As {self.name}, a {self.role}, classify the following query into one of these categories. Just a single classification:
        1. 'local': Can be handled with existing information or {self.role} capabilities. Or requires information about the user that can be obtained from memory.
        2. 'google_calendar': Requires access to or manipulation of Google Calendar data. (Such as adding events, scheduling, updating events. example: "Can you remind me to go grocery shopping this wenesday at 12 pm")

        Query: "{query}"

        Classification (local/google_calendar):
        Explanation:
        """
        response = self.ask([{'role': 'user', 'content': classification_prompt}], temperature=0.3).strip()
        
        classification = 'local'
        explanation = ''
        
        if 'google_calendar' in response.lower():
            classification = 'google_calendar'
       
        explanation_match = re.search(r'Explanation:(.*)', response, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        logger.info(f"Query classification: {classification}")
        logger.info(f"Classification explanation: {explanation}")
        
        return classification, explanation

    def generate_response(self, user_input):
        self.initialize()

        classification, explanation = self.classify_query(user_input)
        
        context = self.memory.get_relevant_context(user_input)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if classification == 'local':
            response = self.perform_local_query(user_input, context, current_time)
        elif classification == 'google_calendar':
            response = self.google_calendar_query(user_input)
        
        self.memory.add_conversation([
            {"role": "user", "content": user_input, "timestamp": current_time},
            {"role": self.role, "content": response, "timestamp": current_time}
        ])

        logger.debug(f"DEBUG: LLM generated response: {response}")
        
        return response

    def extract_parameters(self, function_name, user_input):
        current_time = datetime.now()
        parameter_prompt = f"""
        Extract the necessary parameters for the {function_name} function from the following user query:

        "{user_input}"

        Current date and time: {current_time.strftime("%Y-%m-%d %H:%M:%S")}

        For create_event, extract:
        - summary (event title)
        - start (date and time)
        - end (date and time, if not provided, assume 1 hour after start)
        - description (if any)
        - location (if any)

        For update_event, extract:
        - event_name (current event name)
        - updates (what needs to be changed)

        For delete_event, extract:
        - event_name

        For get_event_details, extract:
        - event_name

        For get_events, extract:
        - start_date (if provided)
        - end_date (if provided)

        Handle various date and time formats, including:
        - Exact dates and times (e.g., "2023-08-15 14:30")
        - Relative dates (e.g., "tomorrow", "next Tuesday", "in 3 days")
        - Time-only formats (e.g., "at 2 PM", "14:00")
        - Date-only formats (e.g., "August 15th", "next Monday")

        If a specific time is not provided, assume 9:00 AM for start times and 5:00 PM for end times.
        If only a time is provided (without a date), assume it's for the current date or the next occurrence of that time.

        Provide the extracted parameters in a Python dictionary format.
        """

        response = self.ask([{'role': 'user', 'content': parameter_prompt}], temperature=0.3).strip()
        
        # Extract the dictionary from the response
        dict_match = re.search(r'\{.*\}', response, re.DOTALL)
        if dict_match:
            try:
                params = eval(dict_match.group())
                return self.process_extracted_parameters(params, current_time)
            except:
                logger.error("Failed to evaluate extracted parameters")
                return {}
        return {}

    def process_extracted_parameters(self, params, current_time):
        def parse_datetime(date_string, default_time=None):
            try:
                # Try parsing with dateutil
                dt = parser.parse(date_string, default=current_time, fuzzy=True)
                
                # Handle relative dates
                if 'tomorrow' in date_string.lower():
                    dt += timedelta(days=1)
                elif 'next' in date_string.lower():
                    if 'week' in date_string.lower():
                        dt += timedelta(weeks=1)
                    elif 'month' in date_string.lower():
                        dt += relativedelta(months=1)
                    else:  # Assume next day of week
                        days_ahead = dt.weekday() - current_time.weekday()
                        if days_ahead <= 0:
                            days_ahead += 7
                        dt = current_time + timedelta(days=days_ahead)
                
                # If only time was provided, use the date from current_time
                if dt.date() == current_time.date() and 'today' not in date_string.lower():
                    if dt.time() < current_time.time():
                        dt += timedelta(days=1)
                
                # If no time was provided, use the default time
                if dt.time() == current_time.time() and default_time:
                    dt = dt.replace(hour=default_time.hour, minute=default_time.minute)
                
                return dt
            except ValueError:
                logger.error(f"Failed to parse date string: {date_string}")
                return None

        # Process start and end times
        if 'start' in params:
            params['start'] = parse_datetime(params['start'], default_time=datetime.min.time().replace(hour=9))
        if 'end' in params:
            params['end'] = parse_datetime(params['end'], default_time=datetime.min.time().replace(hour=17))
        elif 'start' in params:
            # If end time is not provided, set it to 1 hour after start time
            params['end'] = params['start'] + timedelta(hours=1)

        # Process start_date and end_date for get_events
        if 'start_date' in params:
            params['start_date'] = parse_datetime(params['start_date'])
        if 'end_date' in params:
            params['end_date'] = parse_datetime(params['end_date'])

        return params

    def google_calendar_query(self, user_input):
        self.initialize()
        
        function_selection_prompt = f"""As {self.name}, analyze the following user query related to Google Calendar and select the most appropriate function to use. Choose from:

        1. list_calendars(): Lists the primary calendar.
        2. create_event(summary, start, end, description=None, location=None): Creates a new event.
        3. get_events(start_date=None, end_date=None): Retrieves events within a date range.
        4. update_event(event_name, **kwargs): Updates an existing event.
        5. delete_event(event_name): Deletes an event.
        6. get_event_details(event_name): Retrieves details of a specific event.

        Examples:
        - "Show me my calendars" -> list_calendars()
        - "Schedule a team meeting for today at 2 PM" -> create_event()
        - "What events do I have next week?" -> get_events()
        - "Change the time of my dentist appointment to tomorrow at 3 PM" -> update_event()
        - "Remove the team lunch from my calendar" -> delete_event()
        - "What are the details of my project review?" -> get_event_details()

        User query: "{user_input}"

        Respond with ONLY the function name (e.g., 'create_event') and nothing else:
        """

        response = self.ask([{'role': 'user', 'content': function_selection_prompt}], temperature=0.3).strip()
        
        logger.debug(f"Function selection response: {response}")

        function_match = re.search(r'(list_calendars|create_event|get_events|update_event|delete_event|get_event_details)', response, re.IGNORECASE)
        
        if function_match:
            selected_function = function_match.group(1).lower()
            logger.debug(f"Selected function: {selected_function}")

            # Extract parameters based on the selected function
            params = self.extract_parameters(selected_function, user_input)

            logger.debug(f"Extracted parameters: {params}")

            # Call the appropriate Google Calendar function with extracted parameters
            try:
                if selected_function == 'list_calendars':
                    result = self.gcal.list_calendars()
                elif selected_function == 'create_event':
                    start_time = params.get('start', datetime.now())
                    end_time = params.get('end', start_time + timedelta(hours=1))
                    result = self.gcal.create_event(
                        params.get('summary', 'Untitled Event'),
                        start_time,
                        end_time,
                        description=params.get('description'),
                        location=params.get('location')
                    )
                elif selected_function == 'get_events':
                    start_date = params.get('start_date', datetime.now())
                    end_date = params.get('end_date', start_date + timedelta(days=7))
                    result = self.gcal.get_events(start_date, end_date)
                elif selected_function == 'update_event':
                    result = self.gcal.update_event(params.get('event_name', ''), **params.get('updates', {}))
                elif selected_function == 'delete_event':
                    result = self.gcal.delete_event(params.get('event_name', ''))
                elif selected_function == 'get_event_details':
                    result = self.gcal.get_event_details(params.get('event_name', ''))
                else:
                    raise ValueError(f"Unrecognized function: {selected_function}")

                # Use perform_local_query to generate a natural language response
                context = f"Google Calendar function used: {selected_function}\nResult: {result}"
                response = self.perform_local_query(user_input, context, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                return response
            except Exception as e:
                logger.error(f"Error executing Google Calendar function: {str(e)}")
                return f"I'm sorry, but I encountered an error while trying to {selected_function}: {str(e)}. Could you please try rephrasing your request?"
        else:
            logger.error("Failed to extract function from LLM response")
            return "I'm sorry, I couldn't determine the appropriate function to use for your Google Calendar query. Could you please rephrase your request?"

    def perform_local_query(self, user_input, context, current_time):
        prompt = f"""As {self.name}, a {self.role}, respond to the following input:

        Personality:
        {self.personality}

        Context from previous conversations:
        {context}

        Current date and time: {current_time}

        User input:
        {user_input}

        Instructions:
        1. Respond authentically as {self.name}, based on the personality description provided.
        2. Use the context from previous conversations to maintain continuity.
        3. If you don't have enough information to answer the query, state that clearly.
        4. Use the current date and time information when relevant to the query.

        Response:"""

        local_response = self.ask([{'role': 'user', 'content': prompt}], temperature=0.3)
        logger.debug(f"Local response: {local_response}")
        
        return local_response.strip()

# Create a global instance
llm_wrapper = LLMWrapper(os.getenv("LLM_MODEL_PATH"))

if __name__ == "__main__":
    prompt = "Hey schedule a meeting with John for tomorrow at 2 PM" 
    print(f"Processing prompt: {prompt}")
    response = llm_wrapper.generate_response(prompt)
    print("\nGenerated output:")
    print(response)

    print("\nDebug Information:")
    print(f"LLM initialized: {llm_wrapper.llm is not None}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated()}")
    llm_wrapper.close()