from abc import ABC, abstractmethod
import time

class State(ABC):
    '''
    Abstract class representing a runnable state

    A state starts with a status of IDLE after its initialized and before its been run
    When its `start` method is called, it transitions to RUNNING
    Its `run` method needs to be overridden. This is where to place code to be called in a loop
    Its `end` method should be called from within its run method upon completion, which transitions it to FINISHED
    If its `reset` method is called, it transitions back to IDLE

    When overriding `__init__`, `start`, `end`, or `reset`, the respective base class functions should be called
    '''
    IDLE = 0
    RUNNING = 1
    FINISHED = 2

    def __init__(self):
        self._status = State.IDLE

    def start(self):
        self._status = State.RUNNING

    @abstractmethod
    def run(self):
        pass

    def end(self):
        self._status = State.FINISHED
    
    def reset(self):
        '''
        Reset state after it has finished running
        '''
        self._status = State.IDLE
    
    def is_idle(self):
        return self._status == State.IDLE

    def is_running(self):
        return self._status == State.RUNNING
    
    def is_finished(self):
        return self._status == State.FINISHED

class Sequence(State):
    '''
    Runs a given list of states in sequence

    Each state will run to completion (i.e. until its status is IDLE) before the next state starts

    The next state starts running on the next iteration after the previous state finishes
    '''
    def __init__(self, states):
        super().__init__()
        self._states = states
        self._current_state_idx = 0
        self._current_state = states[0]

    def start(self):
        super().start()
        self._current_state.start()

    def run(self):
        if self._current_state.is_idle():
            self._current_state.start()
        if self._current_state.is_running():
            self._current_state.run()
        if self._current_state.is_finished():
            if self._current_state_idx == len(self._states) - 1:
                self.end()
            else:
                self._current_state_idx += 1
                self._current_state = self._states[self._current_state_idx]
    
    def reset(self):
        super().reset()
        for state in self._states:
            state.reset()
        self._current_state_idx = 0
        self._current_state = self._states[0]

class _ParallelStateGroup(State):
    '''
    Runs a given list of states in parallel

    If `wait_for_all` is True, the group ends when all of its constituent states end
    Otherwise, the group ends when one of its constituent states end
    '''
    def __init__(self, states, wait_for_all=True):
        super().__init__()
        self._wait_for_all = wait_for_all
        self._states = states

    def start(self):
        super().start()
        for state in self._states:
            state.start()

    def run(self):
        finished = self._wait_for_all

        for state in self._states:
            if state.is_running():
                if self._wait_for_all:
                    finished = False
                state.run()
            if state.is_finished():
                if not self._wait_for_all:
                    finished = True

        if finished:
            self.end()

    def end(self):
        super().end()
        for state in self._states:
            if not state.is_finished():
                state.end()
    
    def reset(self):
        super().reset()
        for state in self._states:
            state.reset()

class WaitForAll(_ParallelStateGroup):
    '''
    Runs a given list of states in parallel until they are all finished
    '''
    def __init__(self, states):
        super().__init__(states, wait_for_all=True)

class WaitForAny(_ParallelStateGroup):
    '''
    Runs a given list of states in parallel until one of them has finished
    '''
    def __init__(self, states):
        super().__init__(states, wait_for_all=False)

class Repeat(State):
    '''
    Runs a state indefinitely. When it finishes, it is reset and started again
    '''
    def __init__(self, state):
        super().__init__()
        self._state = state
    
    def start(self):
        super().start()
        self._state.start()

    def run(self):
        if self._state.is_idle():
            self._state.start()
        if self._state.is_running():
            self._state.run()
        if self._state.is_finished():
            self._state.reset()
    
    def end(self):
        super().end()
        if not self._state.is_finished():
            self._state.end()
    
    def reset(self):
        super().reset()
        self._state.reset()

class Timer(State):
    '''
    Does nothing for a given duration before finishing
    '''
    def __init__(self, duration):
        '''
        Duration is the amount of time to wait in seconds
        '''
        super().__init__()
        self._duration = duration
        self._start_time = 0
    
    def start(self):
        super().start()
        self._start_time = time.time()
    
    def run(self):
        if time.time() - self._start_time > self._duration:
            self.end()

class Lambda(State):
    '''
    Runs a given function and immediately ends
    '''
    def __init__(self, func):
        super().__init__()
        self._func = func
    
    def run(self):
        self._func()
        self.end()

class InitWrapper(State):
    '''
    Given a state type, initializes the state on its `start` method
    The state is then run normally
    '''
    def __init__(self, state_type, args, kwargs):
        self._state = None
        self._state_args = args
        self._state_kwargs = kwargs
        self._state_type = state_type
    
    def start(self):
        super().start()
        self._state = self._state_type(*self._state_args, **self._state_kwargs)
        self._state.start()
    
    def run(self):
        self._state.run()
    
    def end(self):
        super().end()
        self._state.end()
    
    def reset(self):
        super().reset()
        self._state = None