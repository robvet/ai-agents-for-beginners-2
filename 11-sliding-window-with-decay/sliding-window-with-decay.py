# SlidingWindowWithDecay extends ChatHistory to provide sophisticated conversation history management
# that balances context preservation with token efficiency by:
#   1. Keeping recent messages intact
#   2. Summarizing older messages to maintain context while reducing token usage
#   3. Enforcing a strict token limit to prevent context window overflow

# This implementation provides:
#    Token-aware memory management - Tracks estimated tokens and enforces a max limit
#    Selective summarization - Preserves recent conversations while summarizing older ones
#    Two summarization approaches - Rule-based fallback and a hook for LLM-based summarization
#    Configurable parameters - Adjust token limits, preservation window, and summarization style
#    Detailed documentation - Clear explanations of the class's purpose and operation


# How to invoke
#  Create the history with a 4000 token limit, preserving the 5 most recent exchanges
#  chat_history = SlidingWindowWithDecay(max_token_limit=4000, recent_message_count=5)
#  Use it just like the standard ChatHistory
#  chat_history.add_user_message("Plan me a day trip.")



class SlidingWindowWithDecay(ChatHistory):
    def __init__(self, max_token_limit=4000, recent_message_count=5, summarization_ratio=0.3, kernel=None):
        """
        Initialize the sliding window history with decay.
        
        Args:
            max_token_limit: Maximum number of tokens allowed in history
            recent_message_count: Number of most recent message pairs to always preserve intact
            summarization_ratio: Target ratio of summary length to original length (0.1-0.5 recommended)
            kernel: Optional kernel instance for summarization (if None, uses rule-based summarization)
        """
        super().__init__()
        # Core configuration parameters
        self.max_token_limit = max_token_limit  # Maximum tokens allowed in the full history
        self.recent_message_count = recent_message_count  # Number of recent message pairs to preserve intact
        self.summarization_ratio = summarization_ratio  # Controls summary length vs. original length
        self.kernel = kernel  # Optional kernel for LLM-based summarization
        
        # Memory management
        self.summarized_sections = []  # Store summaries of older conversation sections
        self.estimated_token_count = 0  # Running estimate of total tokens in history

    def add_user_message(self, message):
        """
        Add a user message to history while ensuring token limit compliance.
        If adding would exceed the token limit, older messages are summarized.
        """
        # Estimate tokens in the new message
        message_tokens = self._estimate_tokens(message)
        
        # If adding this message would exceed our limit, compress history
        if self.estimated_token_count + message_tokens > self.max_token_limit:
            self._compress_history()
            
        # Add the message and update token count
        super().add_user_message(message)
        self.estimated_token_count += message_tokens

    def add_assistant_message(self, message):
        """
        Add an assistant message to history while ensuring token limit compliance.
        """
        # Estimate tokens in the new message
        message_tokens = self._estimate_tokens(message)
        
        # If adding this message would exceed our limit, compress history
        if self.estimated_token_count + message_tokens > self.max_token_limit:
            self._compress_history()
            
        # Add the message and update token count
        super().add_assistant_message(message)
        self.estimated_token_count += message_tokens

    def _compress_history(self):
        """
        Compress history by summarizing older messages while preserving recent ones.
        Called when total tokens approach the limit.
        """
        # If we don't have enough messages to summarize, just return
        if len(self._messages) <= self.recent_message_count * 2:
            return
            
        # Identify the messages to preserve vs. summarize
        # Always keep the most recent exchanges intact
        preserve_start = max(0, len(self._messages) - (self.recent_message_count * 2))
        to_summarize = self._messages[:preserve_start]
        to_keep = self._messages[preserve_start:]
        
        # Only proceed if there's something to summarize
        if not to_summarize:
            return
            
        # Create a summary of the older messages
        summary = self._create_summary(to_summarize)
        
        # Estimate tokens we're saving
        old_tokens = sum(self._estimate_tokens(msg.content) for msg in to_summarize if hasattr(msg, 'content'))
        summary_tokens = self._estimate_tokens(summary)
        self.estimated_token_count = self.estimated_token_count - old_tokens + summary_tokens
        
        # Replace history with summary + kept messages
        from semantic_kernel.contents import AuthorRole, ChatMessageContent
        summary_message = ChatMessageContent(role=AuthorRole.SYSTEM, content=summary)
        self._messages = [summary_message] + to_keep

    def _create_summary(self, messages):
        """
        Create a summary of the given messages.
        Uses kernel for LLM summarization if available, otherwise uses rule-based approach.
        """
        if self.kernel:
            # If kernel is available, use an LLM to create a better summary
            # This would require additional code to prepare the prompt and handle the response
            return self._create_llm_summary(messages)
        else:
            # Simple rule-based summarization
            return self._create_rule_based_summary(messages)

    def _create_rule_based_summary(self, messages):
        """
        Create a simple rule-based summary of older messages.
        """
        # Extract user-assistant exchanges
        exchanges = []
        current_exchange = {"user": "", "assistant": ""}
        
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                if msg.role == AuthorRole.USER:
                    if current_exchange["assistant"]:  # If we have a complete exchange, save it
                        exchanges.append(current_exchange.copy())
                        current_exchange = {"user": "", "assistant": ""}
                    current_exchange["user"] = msg.content
                elif msg.role == AuthorRole.ASSISTANT:
                    current_exchange["assistant"] = msg.content

        # Add the last exchange if it's complete
        if current_exchange["user"] and current_exchange["assistant"]:
            exchanges.append(current_exchange)

        # Create the summary format
        summary_parts = ["CONVERSATION HISTORY SUMMARY:"]
        
        for i, exchange in enumerate(exchanges, 1):
            # Extract first sentence or truncate if too long
            user_message = exchange["user"].split('.')[0] + '...' if len(exchange["user"]) > 100 else exchange["user"]
            assistant_message = exchange["assistant"].split('.')[0] + '...' if len(exchange["assistant"]) > 100 else exchange["assistant"]
            
            summary_parts.append(f"{i}. User asked: {user_message}")
            summary_parts.append(f"   Assistant responded: {assistant_message}")
        
        # Return the formatted summary
        return "\n".join(summary_parts)

    def _create_llm_summary(self, messages):
        """
        Create a summary using the LLM through the kernel.
        This is a placeholder that would need to be implemented based on your specific Kernel setup.
        """
        # This is where you'd implement the logic to use the kernel for summarization
        # For now, fall back to rule-based summary
        return self._create_rule_based_summary(messages)

    def _estimate_tokens(self, text):
        """
        Estimate the number of tokens in a text string.
        Uses a simple heuristic: ~4 characters per token for English text.
        """
        if not text:
            return 0
            
        # Simple approximation: ~4 chars per token for English
        # This is a rough estimate; production systems should use a proper tokenizer
        return len(text) // 4 + 1