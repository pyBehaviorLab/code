from . import utility
from . import timer
from . import framework as fw

# State machine variables.

user_task_file = None  # User task definition file module.

states = {}  # Dictionary of {state_name: state_ID}

events = {}  # Dictionary of {event_name: event_ID}

ID2name = {}  # Dictionary of {ID: state_or_event_name}

transition_in_progress = False  # Set to True during state transitions.

variables = None  # User task variables object.

event_dispatch_dict = {}  # {state_name: state behaviour function}

current_state = None
previous_state = None  # Last state visited; used by prev_state nav.

# Optional stage tracking, only populated when the task file defines
# `stage_durations`.
_stage_durations = {}    # {state_name: duration_ms}
_stage_list = []         # Ordered state names that participate in stage navigation.

# State machine functions.


def setup_state_machine(task_file):
    # Initialise the state machine using an imported task definition file.
    global user_task_file, variables, transition_in_progress, states, events, ID2name
    global event_dispatch_dict, _stage_durations, _stage_list, previous_state

    user_task_file = task_file
    variables = utility.v
    transition_in_progress = False
    previous_state = None

    # Assign states and events interger IDs.
    states = {s: i + 1 for s, i in zip(user_task_file.states, range(len(user_task_file.states)))}
    # Reserved intrinsic events, always available without the task declaring
    # them. `zone_changed` fires when a body_part's zone occupancy changes;
    # `frame_event` fires on every pose result (~pose rate) when the operator
    # enables it. Both are host-fired and silent (no TSV row), used like
    # `entry`/`exit`.
    user_events = list(user_task_file.events)
    for _intrinsic in ("zone_changed", "frame_event"):
        if _intrinsic not in user_events:
            user_events.append(_intrinsic)
    events = {
        e: i + 1 + len(user_task_file.states) for e, i in zip(user_events, range(len(user_events)))
    }

    ID2name = {ID: name for name, ID in list(states.items()) + list(events.items())}

    # Make dict mapping state names to state behaviour functions.
    user_task_file_methods = dir(user_task_file)
    for state in list(user_task_file.states) + ["all_states", "run_start", "run_end"]:
        if state in user_task_file_methods:
            event_dispatch_dict[state] = getattr(user_task_file, state)
        else:
            event_dispatch_dict[state] = None

    # Optional stage configuration (auto-advance after entry). Tasks without
    # stage_durations still get pause/resume/next/prev via the timer queue.
    if hasattr(task_file, "stage_durations"):
        _stage_durations = task_file.stage_durations
        s_def = task_file.states
        if isinstance(s_def, (list, tuple)):
            _stage_list = list(s_def)
        else:
            _stage_list = sorted(s_def.keys(), key=lambda s: s_def[s])
    else:
        _stage_durations = {}
        _stage_list = []


def goto_state(next_state):
    # Transition to next state, calling exit action of old state and entry action of next state.
    global transition_in_progress, current_state, previous_state
    if isinstance(next_state, int):  # ID passed in not name.
        next_state = ID2name[next_state]
    if transition_in_progress:
        raise fw.pyControlError("goto_state cannot not be called while processing 'entry' or 'exit' events.")
    if next_state not in states:
        raise fw.pyControlError("Invalid state name passed to goto_state: " + repr(next_state))
    transition_in_progress = True
    process_event("exit")
    timer.disarm_type(fw.STATE_TYP)  # Clear any timed_goto_states
    fw.data_output_queue.put(fw.Datatuple(fw.current_time, fw.STATE_TYP, "", states[next_state]))
    previous_state = current_state
    current_state = next_state
    process_event("entry")
    transition_in_progress = False
    _maybe_schedule_stage_advance()


def process_event(event):
    # Process event given event name by calling appropriate state event handler function.
    if isinstance(event, int):  # ID passed in not name.
        event = ID2name[event]
    if event_dispatch_dict["all_states"]:  # If machine has all_states event handler function.
        handled = event_dispatch_dict["all_states"](event)  # Evaluate all_states event handler function.
        if handled:  # If all_states event handler returns True, don't evaluate state specific behaviour.
            return
    if event_dispatch_dict[current_state]:  # If state machine has event handler function for current state.
        event_dispatch_dict[current_state](event)  # Evaluate state event handler function.


def start():
    global current_state, previous_state
    # Called when run is started. Puts agent in initial state, and runs entry event.
    if event_dispatch_dict["run_start"]:
        event_dispatch_dict["run_start"]()
    previous_state = None
    current_state = user_task_file.initial_state
    fw.data_output_queue.put(fw.Datatuple(fw.current_time, fw.STATE_TYP, "", states[current_state]))
    process_event("entry")
    _maybe_schedule_stage_advance()


def stop():
    # Calls user defined stop function at end of run if function is defined.
    if event_dispatch_dict["run_end"]:
        event_dispatch_dict["run_end"]()


def set_variable(v_name, v_value):
    # Set value of variable v.v_name to v_value.
    try:
        setattr(variables, v_name, v_value)
        return True  # Variable set OK.
    except Exception:
        return False  # Bad variable name or invalid value string.


def get_variable(v_name):
    # Return the value of specified variable.
    try:
        return getattr(variables, v_name)
    except Exception:
        return None  # Bad variable name


# ---------------------------------------------------------------------------
# Optional stage auto-advance (only fires when task defines stage_durations).
# ---------------------------------------------------------------------------


def _maybe_schedule_stage_advance():
    # After a state entry, arm a timed_goto_state to the next stage if the
    # state has a duration. No-op when stage_durations isn't defined.
    if not _stage_durations or current_state not in _stage_list:
        return
    dur = _stage_durations.get(current_state)
    if not dur or dur <= 0:
        return
    idx = _stage_list.index(current_state)
    if idx < len(_stage_list) - 1:
        next_name = _stage_list[idx + 1]
        timer.set(dur, fw.STATE_TYP, "", states[next_name])


# ---------------------------------------------------------------------------
# Navigation API, pause / resume / next / prev. Driven by the host GUI
# via the framework's NAV_TYP byte protocol.
# ---------------------------------------------------------------------------


def _log_nav(text):
    # Surface nav actions in the data stream (live status log + saved file).
    fw.data_output_queue.put(fw.Datatuple(fw.current_time, fw.PRINT_TYP, "t", text))


def pause():
    # Freeze every pending timer (STATE_TYP transitions and EVENT_TYP timers).
    _log_nav("** paused")
    timer.pause_state_timers()
    for event_id in events.values():
        timer.pause(event_id)


def resume():
    # Re-arm every timer captured by pause(), preserving its remaining time.
    _log_nav("** resumed")
    timer.unpause_state_timers()
    for event_id in events.values():
        timer.unpause(event_id)


def next_state():
    # Fire the soonest-pending timed_goto_state immediately. Falls back to
    # the next stage in _stage_list when no STATE_TYP timer is pending.
    target = timer.peek_next_state_target()
    if target is None and _stage_list:
        cur = current_state
        if cur in _stage_list:
            idx = _stage_list.index(cur)
            if idx + 1 < len(_stage_list):
                target = states[_stage_list[idx + 1]]
    if target is None:
        _log_nav("** no scheduled next state")
        return
    _log_nav("** next_state")
    # Drop stale paused STATE_TYP timers; the new state sets up fresh ones.
    timer.paused_timers = [t for t in timer.paused_timers if t.type != fw.STATE_TYP]
    goto_state(target)


def prev_state():
    # Go back to the state we just left. Falls back to the previous stage
    # when previous_state isn't useful (e.g. just started the task).
    target = previous_state
    if target is None and _stage_list:
        cur = current_state
        if cur in _stage_list:
            idx = _stage_list.index(cur)
            if idx > 0:
                target = _stage_list[idx - 1]
    if target is None:
        _log_nav("** no previous state")
        return
    _log_nav("** prev_state")
    timer.paused_timers = [t for t in timer.paused_timers if t.type != fw.STATE_TYP]
    goto_state(target)


# Aliases for task files (and utility.py shims) that call the stage names.
pause_stage = pause
resume_stage = resume
next_stage = next_state
prev_stage = prev_state


def stage_index():
    # 0-based index of current_state in the stage sequence; -1 if not a stage.
    if current_state in _stage_list:
        return _stage_list.index(current_state)
    return -1


def stage_remaining():
    # Milliseconds left on the current stage's auto-advance timer; 0 if none.
    if not _stage_list or current_state not in _stage_list:
        return 0
    idx = _stage_list.index(current_state)
    if idx >= len(_stage_list) - 1:
        return 0
    next_state_id = states[_stage_list[idx + 1]]
    for t in reversed(timer.active_timers):
        if t.type == fw.STATE_TYP and t.content == next_state_id:
            return t.time - fw.current_time
    return 0


def stage_count():
    # Total number of stages in the current task's sequence.
    return len(_stage_list)


def is_stage_active():
    # True when the task defined stage_durations.
    return bool(_stage_list)
