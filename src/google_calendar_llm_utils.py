import os
import re
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.relativedelta import relativedelta
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def extract_parameters(ask_function, function_name, user_input):
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

    response = ask_function([{'role': 'user', 'content': parameter_prompt}], temperature=0.3).strip()
    
    # Extract the dictionary from the response
    dict_match = re.search(r'\{.*\}', response, re.DOTALL)
    if dict_match:
        try:
            params = eval(dict_match.group())
            return process_extracted_parameters(params, current_time)
        except:
            logger.error("Failed to evaluate extracted parameters")
            return {}
    return {}

def process_extracted_parameters(params, current_time):
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
        params = extract_parameters(self.ask, selected_function, user_input)

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