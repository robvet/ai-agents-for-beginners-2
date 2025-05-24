# Implementing a custom chat history class to manage conversation state
# Provide precise control over how messages are stored and retrieved
# This is useful for maintaining context, managing memory, and ensuring 
# # the agent can respond appropriately to user queries.
class SlidingWindowHistory(ChatHistory):    
    def __init__(self, window_size=10, decay_factor=0.8):
        super().__init__()
        self.window_size = window_size
        self.decay_factor = decay_factor
        
    def add_user_message(self, message):
        # Implement windowing logic - remove oldest if past window size
        if len(self._messages) >= self.window_size * 2:  # Account for pairs of messages
            # Remove oldest or create summary/decay of old messages
            self._apply_decay()
            
        # Then add the new message
        super().add_user_message(message)
        
    def _apply_decay(self):
        # Example implementation - could summarize oldest messages
        # or remove them entirely based on your strategy
        oldest = self._messages[0:2]  # Get oldest exchange
        summary = f"Earlier: {oldest[1].content[:50]}..."
        self._messages = self._messages[2:]  # Remove oldest
        # Optionally insert summary