
import unittest
import datetime
from chronos_bot import ChronosEngine, Signal, config

class TestChronosBot(unittest.TestCase):
    def setUp(self):
        # Reset counters implicitly by creating fresh bot
        pass
        
    def test_ny_orb_trigger(self):
        """Test if 09:45 triggers C1."""
        target_time = datetime.datetime(2025, 1, 1, 9, 45, 0)
        
        bot = ChronosEngine(time_source=lambda: target_time)
        
        # Override execute_trade to capture signal instead of logging
        output_signal = None
        def mock_execute(sig):
            nonlocal output_signal
            output_signal = sig
            
        bot.execute_trade = mock_execute
        
        # Run single step
        bot.run(single_step=True)
        
        self.assertIsNotNone(output_signal)
        self.assertEqual(output_signal.concept, "C1_NY_ORB")
        print("PASS: NY ORB Triggered correctly.")

    def test_3pm_macro_trigger(self):
        """Test if 15:00 triggers C3/C8."""
        target_time = datetime.datetime(2025, 1, 1, 15, 0, 0) # 3:00 PM
        
        bot = ChronosEngine(time_source=lambda: target_time)
        
        signals = []
        def mock_execute(sig):
            signals.append(sig)
        bot.execute_trade = mock_execute
        
        bot.run(single_step=True)
        
        # 15:00 triggers C3 and C8.
        # Check if we got both.
        self.assertTrue(len(signals) >= 2)
        
        concepts = [s.concept for s in signals]
        self.assertIn("C3_3PM_MACRO", concepts)
        self.assertIn("C8_LAST_HOUR_MOMENTUM", concepts)
        
        print(f"PASS: 15:00 Triggered Multiple: {concepts}")

    def test_silver_bullet_trigger(self):
        """Test if 10:15 triggers C14."""
        target_time = datetime.datetime(2025, 1, 1, 10, 15, 0)
        
        bot = ChronosEngine(time_source=lambda: target_time)
        
        # Override execute
        signals = []
        bot.execute_trade = lambda s: signals.append(s)
        
        bot.run(single_step=True)
        
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].concept, "C14_SILVER_BULLET")
        print("PASS: 10:15 Silver Bullet Triggered.")
        
    def test_no_trigger_time(self):
        """Test if 10:00 (non-trigger) does nothing."""
        target_time = datetime.datetime(2025, 1, 1, 10, 0, 0)
        bot = ChronosEngine(time_source=lambda: target_time)
        
        signals = []
        bot.execute_trade = lambda s: signals.append(s)
        
        bot.run(single_step=True)
        self.assertEqual(len(signals), 0)
        print("PASS: 10:00 Silent.")

if __name__ == "__main__":
    unittest.main()
