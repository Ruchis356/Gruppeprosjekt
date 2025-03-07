import unittest
import sys, os

# Add the parent directory of 'notebooks' to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from DeckOfCards import DeckOfCards


#Check if the deck is created with 52 cards
class TestDeckOfCards(unittest.TestCase):
    def test_deck_init(self):
        deck=DeckOfCards()
        self.assertEqual(len(deck.cards), 52) 

#Check if the hand is dealt 5 cards
    def test_deal_hand(self):
        deck = DeckOfCards()
        hand = deck.deal_hand(5)
        self.assertEqual(len(hand.cards), 5)

#Check if dealing more cards than are in the deck raises an error
    def test_deal_hand_too_many_cards(self):
        deck = DeckOfCards()
        with self.assertRaises(ValueError):
            deck.deal_hand(53)

#Check to see if all the cards dealt are unique
    def test_deal_hand_unique_cards(self):
        deck = DeckOfCards()
        hand = deck.deal_hand(5)
        unique_cards = set(hand.cards)
        self.assertEqual(len(unique_cards), len(hand.cards))

#Check that the hand dealt is a string representation, and not empty
    def test_str_representation(self):
        deck = DeckOfCards()
        deck_str = str(deck)
        self.assertIsInstance(deck_str, str) 
        self.assertGreater(len(deck_str), 0)

if __name__ == "__main__":
    unittest.main()











    ****






import unittest
import sys, os

# Add the parent directory of 'notebooks' to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from HandOfCards import HandOfCards
from PlayingCard import PlayingCard

#A test-set of cards
cards = [
    PlayingCard("H", 10),  # 10 of Hearts
    PlayingCard("D", 11),   # Jack of Diamonds
    PlayingCard("S", 12),   # Queen of Spades
    PlayingCard("C", 13),   # King of Clubs
    PlayingCard("H", 1)    # Ace of Hearts
    ]

#Test that the hand is initialized with a correct number of cards, correctly
class TestHandOfCards(unittest.TestCase):
    def test_hand_initialization(self):
        hand = HandOfCards(cards)
        self.assertEqual(len(hand.cards), 5)

#Test that it actually catches a flush
    def test_is_flush_true(self):
        flush_cards = [
            PlayingCard("H", 10),  # 10 of Hearts
            PlayingCard("H", 11),   # Jack of Hearts
            PlayingCard("H", 12),   # Queen of Hearts
            PlayingCard("H", 13),   # King of Hearts
            PlayingCard("H", 1)    # Ace of Hearts
        ]
        hand = HandOfCards(flush_cards)
        self.assertTrue(hand.is_flush()) 

#Test that it returns false when it is not a flush
    def test_is_flush_false(self):
        hand = HandOfCards(cards)
        self.assertFalse(hand.is_flush()) 

#Test for the "return Heart cards" function to see if it returns the correct cards       
    def test_is_hearts_with_hearts(self):
        hand = HandOfCards(cards)
        self.assertEqual(hand.is_hearts(), "H10 H1")

#Test for a False result when there are no Hearts
    def test_is_hearts_no_hearts(self):
        no_heart_cards = [
            PlayingCard("C", 10),  # 10 of Hearts
            PlayingCard("D", 11),   # Jack of Diamonds
            PlayingCard("S", 12),   # Queen of Spades
            PlayingCard("C", 13),   # King of Clubs
            PlayingCard("D", 1)    # Ace of Hearts
            ]
        hand = HandOfCards(no_heart_cards)
        self.assertFalse(hand.is_hearts()) 

#Test that it counts the points correctly
    def test_count_points(self):
        hand = HandOfCards(cards)
        self.assertEqual(hand.count_points(), 47)

#Test that it correctly identifies the Queen of spades
    def test_is_ladyspade_true(self):
        hand = HandOfCards(cards)
        self.assertTrue(hand.is_ladyspade())

#Test that it correctly identifies the lack of the Queen of spades
    def test_is_ladyspade_false(self):
        cards = [
            PlayingCard("H", 10),  # 10 of Hearts
            PlayingCard("D", 11),   # Jack of Diamonds
            PlayingCard("D", 12),   # Queen of Spades
            PlayingCard("C", 13),   # King of Clubs
            PlayingCard("H", 1)    # Ace of Hearts
            ]
        hand = HandOfCards(cards)
        self.assertFalse(hand.is_ladyspade()) 

#Test that the string representation of the hand is ok
    def test_str_representation(self):
        hand = HandOfCards(cards)
        self.assertEqual(str(hand), "H10, D11, S12, C13, H1")  # Check the string representation

if __name__ == "__main__":
    unittest.main()






    ****







import unittest
import sys, os

# Add the parent directory of 'notebooks' to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PlayingCard import PlayingCard


class TestPlayingCard(unittest.TestCase):

    #Test that a card is initialized correctly when given valid inputs
    def test_valid_initialization(self):
        card = PlayingCard("H", 10)  # 10 of Hearts
        self.assertEqual(card.get_suit(), "H")
        self.assertEqual(card.get_face(), 10)

    #Test that an invalid suid creates an error
    def test_invalid_suit_initialization(self):
        with self.assertRaises(ValueError):
            PlayingCard("X", 10)  # Invalid suit

    #Test that an invalid number creates an error
    def test_invalid_face_initialization(self):
        with self.assertRaises(ValueError):
            PlayingCard("H", 14)  # Invalid face

    #Test that it strings correctly
    def test_get_as_string(self):
        card = PlayingCard("H", 10)  # 10 of Hearts
        self.assertEqual(card.get_as_string(), "H10")

    #Test that get_suit() returns the correc suit
    def test_get_suit(self):
        card = PlayingCard("D", 5)  # 5 of Diamonds
        self.assertEqual(card.get_suit(), "D")

    #Test that get_face() returns the correct value
    def test_get_face(self):
        card = PlayingCard("S", 12)  # Queen of Spades
        self.assertEqual(card.get_face(), 12)

    #Testing two identical cards are in fact equal
    def test_equality(self):
        card1 = PlayingCard("H", 10)  # 10 of Hearts
        card2 = PlayingCard("H", 10)  # 10 of Hearts
        card3 = PlayingCard("D", 10)  # 10 of Diamonds
        self.assertEqual(card1, card2)  # Same suit and face
        self.assertNotEqual(card1, card3)  # Different suit

    #Testing the hash of a card
    def test_hash(self):
        card1 = PlayingCard("H", 10)  # 10 of Hearts
        card2 = PlayingCard("H", 10)  # 10 of Hearts
        card3 = PlayingCard("D", 10)  # 10 of Diamonds
        self.assertEqual(hash(card1), hash(card2))  # Same suit and face
        self.assertNotEqual(hash(card1), hash(card3))  # Different suit

if __name__ == "__main__":
    unittest.main()