"""
Selenium Integration for Bug Prediction System
Automated web testing and metrics collection
"""

import time
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import selenium, but provide helpful message if not installed
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    from selenium.webdriver.chrome.service import Service
    SELENIUM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Selenium not installed: {e}")
    print("Install with: pip install selenium webdriver-manager")
    SELENIUM_AVAILABLE = False

class SeleniumBugPredictor:
    """Integration of Selenium with Bug Prediction System"""
    
    def __init__(self, browser='chrome', headless=True):
        """
        Initialize Selenium WebDriver
        
        Parameters:
        browser (str): 'chrome', 'firefox', or 'edge'
        headless (bool): Run browser in headless mode
        """
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is not installed. Run: pip install selenium webdriver-manager")
        
        self.browser = browser
        self.headless = headless
        self.driver = None
        self.metrics = []
        self.setup_driver()
        
    def setup_driver(self):
        """Setup Selenium WebDriver"""
        try:
            if self.browser.lower() == 'chrome':
                options = webdriver.ChromeOptions()
                if self.headless:
                    options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                options.add_argument('--disable-blink-features=AutomationControlled')
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_experimental_option('useAutomationExtension', False)
                
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
                
            elif self.browser.lower() == 'firefox':
                options = webdriver.FirefoxOptions()
                if self.headless:
                    options.add_argument('--headless')
                service = Service(GeckoDriverManager().install())
                self.driver = webdriver.Firefox(service=service, options=options)
            else:
                raise ValueError(f"Unsupported browser: {self.browser}")
            
            self.driver.implicitly_wait(10)
            print(f"✓ {self.browser.capitalize()} WebDriver initialized (Headless: {self.headless})")
            
        except Exception as e:
            print(f"❌ Failed to initialize WebDriver: {e}")
            print("\nTroubleshooting tips:")
            print("1. Install Chrome browser: https://www.google.com/chrome/")
            print("2. Run: pip install --upgrade selenium webdriver-manager")
            print("3. Or try Firefox: pip install webdriver-manager")
            self.driver = None
    
    def collect_page_metrics(self, url, page_name):
        """
        Collect metrics from a web page
        
        Parameters:
        url (str): Page URL to analyze
        page_name (str): Name of the page/module
        
        Returns:
        dict: Collected metrics
        """
        if not self.driver:
            print("❌ WebDriver not initialized")
            return None
            
        print(f"📊 Analyzing: {page_name} - {url}")
        
        try:
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            # Collect page load metrics using Navigation Timing API
            try:
                load_time = self.driver.execute_script(
                    "return window.performance.timing.loadEventEnd - window.performance.timing.navigationStart"
                )
            except:
                load_time = 0
            
            # Count DOM elements
            total_elements = len(self.driver.find_elements(By.XPATH, "//*"))
            div_count = len(self.driver.find_elements(By.TAG_NAME, "div"))
            input_count = len(self.driver.find_elements(By.TAG_NAME, "input"))
            button_count = len(self.driver.find_elements(By.TAG_NAME, "button"))
            link_count = len(self.driver.find_elements(By.TAG_NAME, "a"))
            script_count = len(self.driver.find_elements(By.TAG_NAME, "script"))
            
            # Count JavaScript errors
            try:
                logs = self.driver.get_log('browser')
                js_error_count = len([log for log in logs if log['level'] == 'SEVERE'])
            except:
                js_error_count = 0
            
            # Count dynamic elements (Angular/React/Vue attributes)
            dynamic_selectors = ["[data-*]", "[ng-*]", "[v-*]", "[@*]", "[reactid*]"]
            dynamic_elements = 0
            for selector in dynamic_selectors:
                try:
                    dynamic_elements += len(self.driver.find_elements(By.CSS_SELECTOR, selector))
                except:
                    pass
            
            # Count forms and complex elements
            form_count = len(self.driver.find_elements(By.TAG_NAME, "form"))
            iframe_count = len(self.driver.find_elements(By.TAG_NAME, "iframe"))
            
            # Calculate complexity metrics
            page_complexity = (total_elements / 100) + (js_error_count * 10)
            loc_approx = total_elements * 3  # Approximate lines of code
            branch_count = len(self.driver.find_elements(By.CSS_SELECTOR, "button, a, [onclick], [ng-click]"))
            
            # Get page title and meta information
            page_title = self.driver.title
            
            metrics = {
                'module_name': page_name,
                'url': url,
                'title': page_title,
                'loc': min(loc_approx, 10000),  # Cap at 10000
                'v(g)': min(page_complexity, 100),  # Cap at 100
                'branchCount': min(branch_count, 100),
                'e': load_time * 1000 if load_time > 0 else 50000,  # Halstead Effort approximation
                'n': total_elements,
                'total_Op': input_count + button_count,
                'total_Opnd': link_count,
                'uniq_Op': min(len(set([elem.tag_name for elem in self.driver.find_elements(By.XPATH, "//*")])), 50),
                'uniq_Opnd': min(len(set([elem.get_attribute('id') for elem in self.driver.find_elements(By.XPATH, "//*[@id]")])), 50),
                'load_time_ms': load_time,
                'total_elements': total_elements,
                'js_errors': js_error_count,
                'dynamic_elements': dynamic_elements,
                'forms': form_count,
                'iframes': iframe_count,
                'divs': div_count,
                'buttons': button_count,
                'links': link_count
            }
            
            self.metrics.append(metrics)
            print(f"   ✓ Elements: {total_elements}, JS Errors: {js_error_count}, Complexity: {page_complexity:.1f}")
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error analyzing {url}: {e}")
            return None
    
    def test_sample_page(self):
        """Test with a sample page to verify setup"""
        print("\n🧪 Testing with sample pages...")
        
        # Test pages (using public websites for demonstration)
        test_pages = [
            {'name': 'GitHub', 'url': 'https://github.com'},
            {'name': 'Stack Overflow', 'url': 'https://stackoverflow.com'},
            {'name': 'Python.org', 'url': 'https://python.org'}
        ]
        
        results = []
        for page in test_pages:
            metrics = self.collect_page_metrics(page['url'], page['name'])
            if metrics:
                results.append(metrics)
        
        return results
    
    def generate_demo_report(self, metrics_data, output_path='reports'):
        """
        Generate a demo report from collected metrics
        
        Parameters:
        metrics_data (list): List of collected metrics
        output_path (str): Output directory
        """
        os.makedirs(output_path, exist_ok=True)
        
        report = []
        report.append("="*70)
        report.append("SELENIUM WEB TESTING REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Browser: {self.browser}")
        report.append(f"Pages Analyzed: {len(metrics_data)}")
        report.append("")
        
        if metrics_data:
            report.append("PAGE METRICS SUMMARY")
            report.append("-"*40)
            
            for metric in metrics_data:
                report.append(f"\n📄 {metric['module_name']}")
                report.append(f"   URL: {metric['url']}")
                report.append(f"   Title: {metric.get('title', 'N/A')}")
                report.append(f"   Lines of Code (approx): {metric['loc']}")
                report.append(f"   Complexity Score: {metric['v(g)']:.1f}")
                report.append(f"   Branch Count: {metric['branchCount']}")
                report.append(f"   Total Elements: {metric['total_elements']}")
                report.append(f"   JavaScript Errors: {metric['js_errors']}")
                report.append(f"   Dynamic Elements: {metric['dynamic_elements']}")
                report.append(f"   Load Time: {metric['load_time_ms']}ms")
                
                # Risk assessment
                if metric['v(g)'] > 50 or metric['js_errors'] > 5:
                    risk = "🔴 HIGH"
                    action = "Immediate testing required"
                elif metric['v(g)'] > 30 or metric['js_errors'] > 2:
                    risk = "🟡 MEDIUM"
                    action = "Schedule for testing"
                else:
                    risk = "🟢 LOW"
                    action = "Routine QA"
                
                report.append(f"   Risk Level: {risk}")
                report.append(f"   Recommended Action: {action}")
            
            # Summary statistics
            report.append("\n" + "="*70)
            report.append("SUMMARY STATISTICS")
            report.append("-"*40)
            
            avg_complexity = np.mean([m['v(g)'] for m in metrics_data])
            avg_errors = np.mean([m['js_errors'] for m in metrics_data])
            high_risk_count = len([m for m in metrics_data if m['v(g)'] > 50 or m['js_errors'] > 5])
            
            report.append(f"Average Complexity: {avg_complexity:.1f}")
            report.append(f"Average JS Errors: {avg_errors:.1f}")
            report.append(f"High Risk Pages: {high_risk_count}")
            
            # Recommendations
            report.append("\n" + "="*70)
            report.append("RECOMMENDATIONS")
            report.append("-"*40)
            
            if high_risk_count > 0:
                report.append(f"🚨 {high_risk_count} high-risk page(s) detected!")
                report.append("   • Perform immediate security and performance testing")
                report.append("   • Review JavaScript error logs")
                report.append("   • Optimize page complexity")
            else:
                report.append("✓ No high-risk pages detected")
                report.append("   • Continue regular testing schedule")
            
            # Save report
            report_path = os.path.join(output_path, f'selenium_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            
            print(f"\n✓ Report saved to {report_path}")
            
            # Save metrics as JSON
            json_path = os.path.join(output_path, 'page_metrics.json')
            with open(json_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            print(f"✓ Metrics saved to {json_path}")
            
            return report_path
        
        else:
            print("❌ No metrics data to generate report")
            return None
    
    def close(self):
        """Close WebDriver"""
        if self.driver:
            self.driver.quit()
            print("✓ WebDriver closed")

def main():
    """Main function to demonstrate Selenium integration"""
    
    print("="*60)
    print("🧪 SELENIUM WEB TESTING INTEGRATION")
    print("="*60)
    
    if not SELENIUM_AVAILABLE:
        print("\n❌ Selenium is not installed!")
        print("\nPlease install required packages:")
        print("   pip install selenium webdriver-manager")
        print("\nAlso ensure you have Chrome browser installed:")
        print("   https://www.google.com/chrome/")
        return
    
    # Initialize Selenium tester
    try:
        tester = SeleniumBugPredictor(browser='chrome', headless=True)
        
        # Test with sample pages
        metrics = tester.test_sample_page()
        
        # Generate report
        if metrics:
            tester.generate_demo_report(metrics)
        
        # Close driver
        tester.close()
        
        print("\n" + "="*60)
        print("✅ Selenium testing completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Chrome browser is installed")
        print("2. Run: pip install --upgrade selenium webdriver-manager")
        print("3. Try running without headless mode")
        print("4. Check your internet connection")

if __name__ == "__main__":
    main()