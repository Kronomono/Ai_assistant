from gcsa.google_calendar import GoogleCalendar
from gcsa.event import Event
from gcsa.recurrence import Recurrence, DAILY, WEEKLY, MONTHLY
from gcsa.reminders import EmailReminder, PopupReminder
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class GoogleCalendarManager:
    def __init__(self):
        credentials = os.getenv('GOOGLE_CALENDAR_CREDENTIALS')
        self.calendar = GoogleCalendar(credentials_path=credentials)

    def list_calendars(self):
        calendar = self.calendar.get_calendar()
        return f"## Primary Calendar\n\n- {calendar.summary} (ID: {calendar.id})"

    def create_event(self, summary, start, end, description=None, location=None, recurrence=None, reminders=None):
        event = Event(
            summary=summary,
            start=start,
            end=end,
            description=description,
            location=location,
            recurrence=recurrence,
            reminders=reminders
        )
        created_event = self.calendar.add_event(event)
        return f"## Event Created\n\n- Summary: {created_event.summary}\n- Start: {created_event.start}\n- End: {created_event.end}\n- ID: {created_event.id}"

    def get_events(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if end_date is None:
            end_date = start_date + timedelta(days=7)
        
        events = list(self.calendar.get_events(time_min=start_date, time_max=end_date))
        events_list = [f"- {event.summary} (Start: {event.start}, End: {event.end}, ID: {event.id})" for event in events]
        return f"## Events from {start_date} to {end_date}\n\nTotal events: {len(events)}\n\n" + "\n".join(events_list)

    def find_event_by_name(self, event_name, start_date=None, end_date=None):
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if end_date is None:
            end_date = start_date + timedelta(days=30)  # Search within a 30-day window by default
        
        events = list(self.calendar.get_events(time_min=start_date, time_max=end_date))
        matching_events = [event for event in events if event.summary.lower() == event_name.lower()]
        
        if not matching_events:
            return None
        elif len(matching_events) == 1:
            return matching_events[0].id
        else:
            return [event.id for event in matching_events]

    def update_event(self, event_id, **kwargs):
        try:
            event_to_update = self.calendar.get_event(event_id)
            for key, value in kwargs.items():
                setattr(event_to_update, key, value)
            
            updated_event = self.calendar.update_event(event_to_update)
            return f"## Event Updated\n\n- Summary: {updated_event.summary}\n- Start: {updated_event.start}\n- End: {updated_event.end}\n- ID: {updated_event.id}"
        except Exception as e:
            return f"Error updating event: {str(e)}"

    def delete_event(self, event_id):
        try:
            self.calendar.delete_event(event_id)
            return f"Event with ID: {event_id} has been deleted."
        except Exception as e:
            return f"Error deleting event: {str(e)}"

    def get_event_details(self, event_id):
        try:
            event = self.calendar.get_event(event_id)
            return f"## Event Details\n\n- Summary: {event.summary}\n- Start: {event.start}\n- End: {event.end}\n- Description: {event.description}\n- Location: {event.location}\n- ID: {event.id}"
        except Exception as e:
            return f"Error retrieving event details: {str(e)}"

# Usage example:
if __name__ == "__main__":
    gcal = GoogleCalendarManager()
    
    # Create a new event
    start_time = datetime.now() + timedelta(days=1)
    end_time = start_time + timedelta(hours=1)
    print(gcal.create_event("Test Event", start_time, end_time, description="This is a test event"))
    
    # Find the event by name
    event_id = gcal.find_event_by_name("Test Event")
    if event_id:
        # Get event details
        print(gcal.get_event_details(event_id))
        
        # Update the event
        print(gcal.update_event(event_id, description="Updated description"))
        
        # Delete the event
        print(gcal.delete_event(event_id))
    else:
        print("Event not found")