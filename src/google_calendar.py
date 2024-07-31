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
        credentials_path = os.getenv('GOOGLE_CALENDAR_CREDENTIALS_PATH')
        self.calendar = GoogleCalendar(credentials_path=credentials_path)

    def list_calendars(self):
        calendars = self.calendar.get_calendars()
        return [{'id': cal.id, 'summary': cal.summary} for cal in calendars]

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
        return self.calendar.add_event(event)

    def get_events(self, start=None, end=None):
        if not start:
            start = datetime.now()
        if not end:
            end = start + timedelta(days=7)
        events = self.calendar.get_events(start, end)
        return [{'id': event.id, 'summary': event.summary, 'start': event.start, 'end': event.end} for event in events]

    def update_event(self, event_id, **kwargs):
        event = self.calendar.get_event(event_id)
        for key, value in kwargs.items():
            setattr(event, key, value)
        return self.calendar.update_event(event)

    def delete_event(self, event_id):
        event = self.calendar.get_event(event_id)
        self.calendar.delete_event(event)

    def quick_add_event(self, text):
        return self.calendar.quick_add_event(text)

    def set_reminders(self, event_id, reminders):
        event = self.calendar.get_event(event_id)
        event.reminders = reminders
        return self.calendar.update_event(event)

    @staticmethod
    def create_recurrence(freq, **kwargs):
        return Recurrence.rule(freq=freq, **kwargs)

    @staticmethod
    def create_reminder(method, minutes):
        if method.lower() == 'email':
            return EmailReminder(minutes_before_start=minutes)
        elif method.lower() == 'popup':
            return PopupReminder(minutes_before_start=minutes)
        else:
            raise ValueError("Invalid reminder method. Use 'email' or 'popup'.")
        
    def get_events_next_week(self):
            start = datetime.now()
            end = start + timedelta(days=7)
            events = self.get_events(start, end)
            formatted_events = []
            for event in events:
                event_info = f"Event: {event['summary']}\nStart: {event['start']}\nEnd: {event['end']}\n"
                if 'description' in event and event['description']:
                    event_info += f"Description: {event['description']}\n"
                if 'location' in event and event['location']:
                    event_info += f"Location: {event['location']}\n"
                formatted_events.append(event_info)
            return "\n\n".join(formatted_events)

# Usage example:
if __name__ == "__main__":
    gcal = GoogleCalendarManager()
    
    # List calendars
    print(gcal.list_calendars())
    
    # Create an event
    start = datetime.now() + timedelta(days=1)
    end = start + timedelta(hours=1)
    event = gcal.create_event("Test Event", start, end, description="This is a test event")
    print(f"Created event: {event.id}")
    
    # Get events for the next week
    events = gcal.get_events()
    print(f"Upcoming events: {events}")
    
    # Update an event
    gcal.update_event(event.id, summary="Updated Test Event")
    
    # Set reminders
    reminders = [
        gcal.create_reminder('email', 60),
        gcal.create_reminder('popup', 15)
    ]
    gcal.set_reminders(event.id, reminders)
    
    # Delete the event
    gcal.delete_event(event.id)
    print("Event deleted")
