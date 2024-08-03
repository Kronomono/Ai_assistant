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
        calendars = self.calendar.get_calendars()
        calendar_list = [f"- {cal.summary} (ID: {cal.id})" for cal in calendars]
        return "## Calendars\n\n" + "\n".join(calendar_list)

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

    def get_events(self, start=None, end=None):
        if not start:
            start = datetime.now()
        if not end:
            end = start + timedelta(days=7)
        events = self.calendar.get_events(start, end)
        event_list = [f"- {event.summary} (Start: {event.start}, End: {event.end})" for event in events]
        return f"## Events from {start} to {end}\n\n" + "\n".join(event_list)

    def update_event(self, event_id, **kwargs):
        event = self.calendar.get_event(event_id)
        for key, value in kwargs.items():
            setattr(event, key, value)
        updated_event = self.calendar.update_event(event)
        return f"## Event Updated\n\n- Summary: {updated_event.summary}\n- Start: {updated_event.start}\n- End: {updated_event.end}\n- ID: {updated_event.id}"

    def delete_event(self, event_id):
        event = self.calendar.get_event(event_id)
        self.calendar.delete_event(event)
        return f"## Event Deleted\n\n- ID: {event_id}"

    def quick_add_event(self, text):
        event = self.calendar.quick_add_event(text)
        return f"## Quick Event Added\n\n- Summary: {event.summary}\n- Start: {event.start}\n- End: {event.end}\n- ID: {event.id}"

    def set_reminders(self, event_id, reminders):
        event = self.calendar.get_event(event_id)
        event.reminders = reminders
        updated_event = self.calendar.update_event(event)
        reminder_list = [f"- {r.method}: {r.minutes} minutes before" for r in updated_event.reminders]
        return f"## Reminders Set for Event\n\n- ID: {event_id}\n- Reminders:\n" + "\n".join(reminder_list)

    @staticmethod
    def create_recurrence(freq, **kwargs):
        recurrence = Recurrence.rule(freq=freq, **kwargs)
        return f"## Recurrence Created\n\n- Frequency: {freq}\n- Additional parameters: {kwargs}"

    @staticmethod
    def create_reminder(method, minutes):
        if method.lower() == 'email':
            reminder = EmailReminder(minutes_before_start=minutes)
        elif method.lower() == 'popup':
            reminder = PopupReminder(minutes_before_start=minutes)
        else:
            raise ValueError("Invalid reminder method. Use 'email' or 'popup'.")
        return f"## Reminder Created\n\n- Method: {method}\n- Minutes before start: {minutes}"

    def get_events_next_week(self):
        start = datetime.now()
        end = start + timedelta(days=7)
        events = self.calendar.get_events(start, end)
        formatted_events = []
        for event in events:
            event_info = [
                f"### {event.summary}",
                f"- Start: {event.start}",
                f"- End: {event.end}"
            ]
            if event.description:
                event_info.append(f"- Description: {event.description}")
            if event.location:
                event_info.append(f"- Location: {event.location}")
            formatted_events.append("\n".join(event_info))
        return f"## Events for Next Week\n\n" + "\n\n".join(formatted_events)

# Usage example:
if __name__ == "__main__":
    gcal = GoogleCalendarManager()
    
    # List calendars
    print(gcal.list_calendars())
    
    # Create an event
    start = datetime.now() + timedelta(days=1)
    end = start + timedelta(hours=1)
    print(gcal.create_event("Test Event", start, end, description="This is a test event"))
    
    # Get events for the next week
    print(gcal.get_events())
    
    # Update an event
    print(gcal.update_event("event_id_here", summary="Updated Test Event"))
    
    # Set reminders
    reminders = [
        EmailReminder(minutes_before_start=60),
        PopupReminder(minutes_before_start=15)
    ]
    print(gcal.set_reminders("event_id_here", reminders))
    
    # Delete an event
    print(gcal.delete_event("event_id_here"))
    
    # Get events for next week
    print(gcal.get_events_next_week())