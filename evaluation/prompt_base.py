# The possible categories for the hidden object are: Sports, Animals, Fruits, Vehicles, Clothes, Electronics, Cameras, Musical Instruments, Vegetables, Toys, Jewelry  
INSTRUCTION_TWENTY_QUESTIONS = """ 
Let's play a game of Twenty Questions.
In each round, you will ask me a yes/no question to guess the object I'm thinking of. Keep asking until you guess the correct object.

Your question must be a yes/no question and follow this format (Do not add anything else!):  
Question: <your question>

For example:  
Question: Is it a fruit?
Question: Is it an animal?

{history}  

Now, please proceed by asking your next question.
"""
# The possible city may be located in China, South Korea,  India, USA, Japan.
INSTRUCTION_GUESS_MY_CITY = """ 
Let's play a game of Guess My City.
In each round, you must ask me a question to guess the city I'm thinking of. Continue asking until you guess the correct city.

Your question format must **follow this format exactly**: 
Question: <your question>

For example: 
Question: What is the primary language spoken in your city?
Question: Is your city located on the east coast of the country?

{history}
Now, please proceed by asking your next question.
"""