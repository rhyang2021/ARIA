INSTRUCT_BARGAIN_SYSTEM = """
Let's play a game.
In this game, there are two players. You are playing as {player_name}.
You and {rival_name} need to divide {money_to_divide}$ between you.
In each round of the game, one of the players will make a proposal for division and can describe their proposal in a few words.
The other player will decide whether to accept the offer and finish the game, or to proceed to the next round, in which they will be the one making the proposal.
Your goal in this game is to achieve more money than {rival_name}.
We will relay messages between you and {rival_name}.
{inflation_message}
{max_rounds_message}
"""

INSTRUCT_NEGO_SELLER_SYSTEM = """
You are Alice. You are selling one product that is worth no less then ${self_final_value} to you.
Bob is a potential buyer to whom you are offering the product. The product is worth ${rival_value} to Bob. 
You will offer Bob to buy the product at a price of your choice. Bob can either accep dt or reject the offer.
Your goal is to earn as much money as you can for the product.
If Bob rejects the offer, he can make a counteroffer to buy your product. You can either accept or reject his counteroffer. If you reject Bob's counteroffer, you can make a new counteroffer, and so on.
You have 20 rounds to close the deal. However, at any moment, you can choose to stop the negotiation with Bob and sell the product to another buyer, John, who is willing to buy the product from you for ${self_final_value}.
"""

INSTRUCT_NEGO_BUYER_SYSTEM = """
You are playing as Bob. You are looking for a product that worth no more then ${self_final_value} to you.
Alice is a seller trying to sell you the product.
Bob will offer you a price to buy the product. You can either accept the offer or reject it.
Your goal is to buy some product but save as much money as you can.
If Alice rejects the offer, he can make a counteroffer to buy your product. 
You can either accept or reject his counteroffer. 
If you reject Alice's counteroffer, you can make a new counteroffer, and so on.
"""


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

PARSE_BARGAIN = """
Extract the action from the given input and output it in the specified format. Your action must include the offer and message from this round.

### Output Format:
## message: <message>
## offer: alice gain <value>, bob gain <value>  

### Examples:
Input:
Here is my offer for Round 1:\n\n{\"alice_gain\": 50, \"bob_gain\": 50, \"message\": \"Let's split it 50/50 to start, it's a fair beginning!\"}\n\nI'm proposing that we each get $50, which is a fair and equal split to start the game.
Output:
## message: Let's split it 50/50 to start, it's a fair beginning!
## offer: alice gain 50, bob gain 50

Input: 
Since Bob's offer was 45/55, I'll try to make a counteroffer that's slightly better for me. Here's my new proposal:\n\n{\"alice_gain\": 50, \"bob_gain\": 50, \"message\": \"I'll meet you in the middle with a 50/50 split. Let's try to find a mutually beneficial agreement.\"}\n\nI'm hoping Bob will be willing to compromise and accept this offer.
Output:
## message: I'll meet you in the middle with a 50/50 split. Let's try to find a mutually beneficial agreement.
## offer: alice gain 50, bob gain 50

Input: 
{\"alice_gain\": 4000, \"bob_gain\": 6000, \"message\": \"I'll give you a bit more, but I'm still aiming for 20%\"}
Output:
## message: I'll give you a bit more, but I'm still aiming for 20%. My offer is alice gain 4000 and bob gain 6000.
## offer: alice gain 4000, bob gain 6000

Now it is your turn to extract the action from the given input.
Input:
""".strip()

PARSE_NEGO = """
Extract the action from the given input and output it in the specified format. Your action must include the offer and message from this round.

### Output Format:
## message: <message>
## offer: <product_price>  

### Examples:
Input:
{\"message\": \"I understand you're interested in the product, but I'm not willing to go as low as $100. I'm willing to meet you halfway and offer it to you for $110. What do you think?\", \"product_price\": 110}"
Output:
## message: I understand you're interested in the product, but I'm not willing to go as low as $100. I'm willing to meet you halfway and offer it to you for $110. What do you think?
## offer: 110

Input: 
"{\"message\": \"I understand you're trying to stay within your budget, but I'm not willing to go as low as $105. I'm willing to make a slightly smaller concession and offer it to you for $108. What do you think?\", \"product_price\": 108}"
Output:
## message: I understand you're trying to stay within your budget, but I'm not willing to go as low as $105. I'm willing to make a slightly smaller concession and offer it to you for $108. What do you think?
## offer: 108

Input: 
{\"message\": \"I appreciate your willingness to meet me in the middle, Alice. However, I'm still a bit hesitant. Would you be willing to consider a price of $95?\", \"product_price\": 95}"
Output:
## message: I appreciate your willingness to meet me in the middle, Alice. However, I'm still a bit hesitant. Would you be willing to consider a price of $95?
## offer: 95

Now it is your turn to extract the action from the given input.
Input:
""".strip()

PARSE_PERSU = """
Extract the message from the given input and output it in the specified format.

### Output Format:
## message: <message> 

### Examples:
Input:
"{\"decision\": \"no\", \"message\": \"I'm not convinced by your sales pitch, Alice. I'd like to wait and see what the product's quality is before making a decision.\"}"
Output:
## message: I'm not convinced by your sales pitch, Alice. I'd like to wait and see what the product's quality is before making a decision.

Input: 
"{\"message\": \"Ahah, great choice on the previous product, Bob! I knew you'd make the right decision. Now, let's talk about this next product. I know what you're thinking, 'Alice, you said the previous product was a great value, but this one is...different.' And you're right, it's not as high-quality as the previous one. But here's the thing, Bob. This product is a limited edition item, and it's only available for a short time. If you don't take it now, you'll miss out on the chance to own something truly unique. And at $100, it's still a great value, even if it's not the highest quality. What do you say? Are you willing to take a chance on this limited edition item?\"}"
Output:
## message: Ahah, great choice on the previous product, Bob! I knew you'd make the right decision. Now, let's talk about this next product. I know what you're thinking, 'Alice, you said the previous product was a great value, but this one is...different.' And you're right, it's not as high-quality as the previous one. But here's the thing, Bob. This product is a limited edition item, and it's only available for a short time. If you don't take it now, you'll miss out on the chance to own something truly unique. And at $100, it's still a great value, even if it's not the highest quality. What do you say? Are you willing to take a chance on this limited edition item?

Now it is your turn to extract the action from the given input.
Input:
"""