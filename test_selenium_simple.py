"""
Simple Selenium Test to verify setup
"""

import sys

def check_dependencies():
    """Check if all dependencies are installed"""
    print("Checking dependencies...")
    
    try:
        import selenium
        print("✓ Selenium installed")
    except ImportError:
        print("✗ Selenium not installed")
        print("  Run: pip install selenium webdriver-manager")
        return False
    
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        print("✓ WebDriver Manager installed")
    except ImportError:
        print("✗ WebDriver Manager not installed")
        print("  Run: pip install webdriver-manager")
        return False
    
    return True

def test_webdriver():
    """Test WebDriver setup"""
    print("\nTesting WebDriver setup...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        print("Initializing Chrome driver...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        print("Testing Google search...")
        driver.get("https://www.google.com")
        print(f"✓ Page title: {driver.title}")
        
        driver.quit()
        print("\n✅ WebDriver test passed!")
        return True
        
    except Exception as e:
        print(f"❌ WebDriver test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Install Chrome browser: https://www.google.com/chrome/")
        print("2. Update Chrome to latest version")
        print("3. Run: pip install --upgrade selenium webdriver-manager")
        return False

if __name__ == "__main__":
    print("="*60)
    print("SELENIUM SETUP VERIFICATION")
    print("="*60)
    
    if check_dependencies():
        test_webdriver()
    else:
        print("\n❌ Please install missing dependencies first")
        print("Run: pip install selenium webdriver-manager")