import streamlit as st
import PyPDF2
import docx
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

@dataclass
class TrafficLightResult:
    category: str
    status: str
    score: int
    factors: List[str]
    recommendations: List[str]

@dataclass
class EnrollmentImpact:
    baseline_months: float
    improved_months: float
    time_saved_months: float
    time_saved_percent: float
    cost_savings_estimate: str
    key_drivers: List[str]

@dataclass
class ProductSalesImpact:
    is_new_drug: bool
    peak_annual_sales: float
    daily_sales: float
    time_saved_days: int
    direct_sales_gain: float
    npv_sales_gain: float
    patent_life_extension_value: float
    total_value: float
    assumptions: Dict[str, any]
    market_size_category: str

class ProtocolAnalyzer:
    def __init__(self, text: str):
        self.text = text.lower()
        self.original_text = text  # Keep original case for better detection
        self.word_count = len(text.split())
        self.phase = self._detect_phase()
        self.indication = self._detect_indication()
        self.patient_population_size = self._estimate_population_size()
        self.is_new_drug = self._detect_new_drug()
        self.market_characteristics = self._analyze_market()
    
    def _detect_phase(self) -> str:
        """Detect the study phase with improved Roman numeral detection"""
        
        # Phase detection patterns - ordered from most specific to least specific
        phase_patterns = [
            # Roman numerals with "Phase" keyword
            (r'\bphase\s*I\b(?!\s*[IV])', 'Phase 1', 1),  # Phase I (not followed by V)
            (r'\bphase\s*II\b(?!\s*I)', 'Phase 2', 2),    # Phase II (not followed by I)
            (r'\bphase\s*III\b', 'Phase 3', 3),           # Phase III
            (r'\bphase\s*IV\b', 'Phase 4', 4),            # Phase IV
            
            # Roman numerals with hyphen
            (r'\bphase-I\b(?!\s*[IV])', 'Phase 1', 1),
            (r'\bphase-II\b(?!\s*I)', 'Phase 2', 2),
            (r'\bphase-III\b', 'Phase 3', 3),
            (r'\bphase-IV\b', 'Phase 4', 4),
            
            # Numeric with "Phase" keyword
            (r'\bphase\s*1\b', 'Phase 1', 1),
            (r'\bphase\s*2\b', 'Phase 2', 2),
            (r'\bphase\s*3\b', 'Phase 3', 3),
            (r'\bphase\s*4\b', 'Phase 4', 4),
            
            # With forward slash
            (r'\bphase\s*/\s*I\b(?!\s*[IV])', 'Phase 1', 1),
            (r'\bphase\s*/\s*II\b(?!\s*I)', 'Phase 2', 2),
            (r'\bphase\s*/\s*III\b', 'Phase 3', 3),
            (r'\bphase\s*/\s*IV\b', 'Phase 4', 4),
            
            # Standalone Roman numerals in context
            (r'\bphase\s*:\s*I\b(?!\s*[IV])', 'Phase 1', 1),
            (r'\bphase\s*:\s*II\b(?!\s*I)', 'Phase 2', 2),
            (r'\bphase\s*:\s*III\b', 'Phase 3', 3),
            (r'\bphase\s*:\s*IV\b', 'Phase 4', 4),
            
            # Combined phases (take the highest)
            (r'\bphase\s*I\s*/\s*II\b', 'Phase 1/2', 1.5),
            (r'\bphase\s*II\s*/\s*III\b', 'Phase 2/3', 2.5),
            (r'\bphase\s*1\s*/\s*2\b', 'Phase 1/2', 1.5),
            (r'\bphase\s*2\s*/\s*3\b', 'Phase 2/3', 2.5),
            
            # Written out
            (r'\bphase\s*one\b', 'Phase 1', 1),
            (r'\bphase\s*two\b', 'Phase 2', 2),
            (r'\bphase\s*three\b', 'Phase 3', 3),
            (r'\bphase\s*four\b', 'Phase 4', 4),
        ]
        
        detected_phases = []
        
        # Search in original text (preserves Roman numerals case)
        for pattern, phase_name, priority in phase_patterns:
            if re.search(pattern, self.original_text, re.IGNORECASE):
                detected_phases.append((phase_name, priority))
        
        # Also search in lowercase text for additional patterns
        for pattern, phase_name, priority in phase_patterns:
            if re.search(pattern, self.text):
                detected_phases.append((phase_name, priority))
        
        if detected_phases:
            # Remove duplicates and sort by priority (higher phase = higher priority)
            unique_phases = list(set(detected_phases))
            # Take the highest priority (usually the highest phase number found)
            best_match = max(unique_phases, key=lambda x: x[1])
            return best_match[0]
        
        # Additional context clues if no explicit phase found
        if 'first-in-human' in self.text or 'fih' in self.text:
            return 'Phase 1'
        if 'proof of concept' in self.text or 'dose ranging' in self.text:
            return 'Phase 2'
        if 'pivotal' in self.text or 'confirmatory' in self.text:
            return 'Phase 3'
        if 'post-marketing' in self.text or 'post-approval' in self.text:
            return 'Phase 4'
        
        # Check for regulatory context
        if 'ind' in self.text and 'amendment' not in self.text:
            return 'Phase 1'  # IND typically associated with early phase
        if 'nda' in self.text or 'bla' in self.text:
            return 'Phase 3'  # Near NDA/BLA submission
        
        return 'Phase Not Specified'
    
    def _detect_new_drug(self) -> bool:
        """Detect if this is a new drug application vs. other types"""
        # Indicators of new drug
        new_drug_indicators = [
            'investigational',
            'new molecular entity',
            'nme',
            'first-in-human',
            'fih',
            'novel',
            'new drug application',
            'nda',
            'biologics license application',
            'bla',
            'investigational new drug',
            'ind',
            'first in man',
            'new chemical entity',
            'nce'
        ]
        
        # Indicators it's NOT a new drug (modifications, generics, etc.)
        not_new_drug = [
            'generic',
            'biosimilar',
            'post-marketing',
            'post-approval',
            'label expansion',
            'supplemental',
            'snda',
            'line extension',
            'reformulation'
        ]
        
        # Check for not-new-drug indicators first
        for indicator in not_new_drug:
            if indicator in self.text:
                return False
        
        # Check for new drug indicators
        for indicator in new_drug_indicators:
            if indicator in self.text:
                return True
        
        # Default based on phase
        # Phase 1-3 are typically new drugs unless otherwise indicated
        if self.phase in ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 1/2', 'Phase 2/3']:
            return True
        elif self.phase == 'Phase 4':
            return False  # Phase 4 is post-approval
        
        # If phase not specified, look for other clues
        if 'healthy volunteer' in self.text or 'safety' in self.text and 'tolerability' in self.text:
            return True
        
        return False  # Default to false if uncertain
    
    def _analyze_market(self) -> Dict[str, any]:
        """Analyze market characteristics for sales forecasting"""
        indication = self.indication
        
        # Market size estimates by therapeutic area (annual peak sales potential)
        # Based on industry benchmarks and real-world blockbusters
        market_sizes = {
            'oncology': {
                'blockbuster': 5000000000,  # $5B (e.g., Keytruda, Revlimid)
                'major': 1500000000,         # $1.5B
                'niche': 500000000           # $500M
            },
            'cardiology': {
                'blockbuster': 4000000000,  # $4B (e.g., Eliquis)
                'major': 1200000000,
                'niche': 400000000
            },
            'neurology': {
                'blockbuster': 3500000000,  # $3.5B (e.g., Gilenya for MS)
                'major': 1000000000,
                'niche': 350000000
            },
            'immunology': {
                'blockbuster': 6000000000,  # $6B (e.g., Humira, Stelara)
                'major': 2000000000,
                'niche': 600000000
            },
            'rare_disease': {
                'blockbuster': 2000000000,  # $2B (e.g., Spinraza)
                'major': 800000000,
                'niche': 250000000
            },
            'respiratory': {
                'blockbuster': 3000000000,  # $3B (e.g., Symbicort)
                'major': 900000000,
                'niche': 300000000
            },
            'endocrinology': {
                'blockbuster': 3500000000,  # $3.5B (e.g., Ozempic)
                'major': 1100000000,
                'niche': 350000000
            },
            'infectious_disease': {
                'blockbuster': 2500000000,  # $2.5B (e.g., Biktarvy for HIV)
                'major': 800000000,
                'niche': 250000000
            },
            'psychiatry': {
                'blockbuster': 3000000000,  # $3B
                'major': 1000000000,
                'niche': 300000000
            },
            'general': {
                'blockbuster': 3000000000,
                'major': 1000000000,
                'niche': 300000000
            }
        }
        
        therapeutic_area = indication['therapeutic_area']
        
        # Determine market size category based on various factors
        market_category = 'niche'  # Default
        
        if indication['is_rare']:
            market_category = 'niche'
        elif 'first-line' in self.text or 'standard of care' in self.text or 'frontline' in self.text:
            market_category = 'blockbuster'
        elif self.phase in ['Phase 3', 'Phase 2/3']:
            market_category = 'major'
        elif 'breakthrough' in self.text or 'fast track' in self.text:
            market_category = 'major'
        
        # Specific indication boosts
        if any(term in self.text for term in ['metastatic', 'advanced', 'refractory']):
            if market_category == 'niche':
                market_category = 'major'
        
        peak_sales = market_sizes.get(therapeutic_area, market_sizes['general'])[market_category]
        
        # Orphan drug premium (if both rare AND has orphan designation)
        if indication['is_rare'] and 'orphan' in self.text:
            peak_sales *= 1.5
        
        return {
            'therapeutic_area': therapeutic_area,
            'market_category': market_category,
            'peak_annual_sales': peak_sales,
            'is_rare': indication['is_rare']
        }
    
    def calculate_product_sales_impact(self, enrollment_impact: EnrollmentImpact) -> ProductSalesImpact:
        """Calculate additional product sales from accelerated timeline"""
        
        if not self.is_new_drug:
            return ProductSalesImpact(
                is_new_drug=False,
                peak_annual_sales=0,
                daily_sales=0,
                time_saved_days=0,
                direct_sales_gain=0,
                npv_sales_gain=0,
                patent_life_extension_value=0,
                total_value=0,
                assumptions={},
                market_size_category='N/A'
            )
        
        # Key assumptions - explicitly defined
        assumptions = {
            'peak_annual_sales': self.market_characteristics['peak_annual_sales'],
            'time_to_peak_years': 5,  # Industry standard: 5 years from launch to peak
            'discount_rate': 0.10,  # 10% discount rate (pharma industry standard)
            'peak_sales_duration_years': 5,  # Years at peak before patent cliff
            'patent_life_remaining_years': 12,  # Typical remaining patent life at Phase 3
            'ramp_up_curve': 'S-curve',  # Sales follow S-curve adoption pattern
            'sales_year_1_pct': 0.15,  # 15% of peak in year 1 post-launch
            'sales_year_2_pct': 0.35,  # 35% of peak in year 2
            'sales_year_3_pct': 0.60,  # 60% of peak in year 3
            'sales_year_4_pct': 0.85,  # 85% of peak in year 4
            'sales_year_5_pct': 1.00,  # 100% of peak in year 5
            'market_size_category': self.market_characteristics['market_category'],
            'therapeutic_area': self.market_characteristics['therapeutic_area']
        }
        
        peak_annual_sales = assumptions['peak_annual_sales']
        time_saved_days = int(enrollment_impact.time_saved_months * 30.44)  # Average days per month
        time_saved_years = enrollment_impact.time_saved_months / 12
        
        # EXPLICIT ASSUMPTION: Calculate daily sales at peak
        # Formula: Peak Annual Sales ÷ 365 days
        daily_sales_at_peak = peak_annual_sales / 365
        
        # Method 1: Direct sales gain (simple calculation)
        # EXPLICIT ASSUMPTION: All days saved translate to peak-level sales
        # Formula: Time Saved (days) × Daily Sales at Peak
        # Note: This is conservative as it assumes immediate peak sales
        direct_sales_gain = daily_sales_at_peak * time_saved_days
        
        # Method 2: NPV calculation (more sophisticated and accurate)
        # Accounts for time value of money and realistic sales ramp
        
        def calculate_sales_schedule(years_to_launch, peak_sales, assumptions):
            """
            Calculate year-by-year sales schedule
            
            EXPLICIT ASSUMPTIONS:
            - Launch occurs at years_to_launch
            - Sales ramp follows S-curve over 5 years
            - Peak sales maintained for 5 years
            - Patent cliff causes 30%, 50%, 70% decline over 3 years
            - All cash flows discounted at 10% annually
            """
            sales_schedule = []
            discount_rate = assumptions['discount_rate']
            
            # Ramp-up percentages (based on industry adoption curves)
            ramp_percentages = [
                assumptions['sales_year_1_pct'],  # Year 1: 15%
                assumptions['sales_year_2_pct'],  # Year 2: 35%
                assumptions['sales_year_3_pct'],  # Year 3: 60%
                assumptions['sales_year_4_pct'],  # Year 4: 85%
                assumptions['sales_year_5_pct']   # Year 5: 100%
            ]
            
            # Years before launch: $0 sales
            for year in range(int(years_to_launch)):
                sales_schedule.append({
                    'year': year + 1,
                    'sales': 0,
                    'discount_factor': 1 / ((1 + discount_rate) ** year),
                    'npv': 0
                })
            
            # Ramp-up years (Years 1-5 post-launch)
            launch_year = int(years_to_launch)
            for i, pct in enumerate(ramp_percentages):
                year = launch_year + i
                annual_sales = peak_sales * pct
                discount_factor = 1 / ((1 + discount_rate) ** year)
                sales_schedule.append({
                    'year': year + 1,
                    'sales': annual_sales,
                    'discount_factor': discount_factor,
                    'npv': annual_sales * discount_factor
                })
            
            # Peak years (5 years at full peak)
            peak_duration = assumptions['peak_sales_duration_years']
            peak_start_year = launch_year + 5
            for i in range(peak_duration):
                year = peak_start_year + i
                discount_factor = 1 / ((1 + discount_rate) ** year)
                sales_schedule.append({
                    'year': year + 1,
                    'sales': peak_sales,
                    'discount_factor': discount_factor,
                    'npv': peak_sales * discount_factor
                })
            
            # Decline years (patent cliff - simplified linear decline)
            # ASSUMPTION: 70% → 50% → 30% of peak over 3 years
            decline_years = 3
            decline_start_year = peak_start_year + peak_duration
            for i in range(decline_years):
                year = decline_start_year + i
                annual_sales = peak_sales * (0.7 - i * 0.2)  # 70%, 50%, 30%
                discount_factor = 1 / ((1 + discount_rate) ** year)
                sales_schedule.append({
                    'year': year + 1,
                    'sales': annual_sales,
                    'discount_factor': discount_factor,
                    'npv': annual_sales * discount_factor
                })
            
            return sales_schedule
        
        # Calculate baseline scenario (current timeline)
        baseline_years_to_launch = 0  # Relative to baseline
        baseline_schedule = calculate_sales_schedule(baseline_years_to_launch, peak_annual_sales, assumptions)
        baseline_npv = sum(year['npv'] for year in baseline_schedule)
        
        # Calculate accelerated scenario (earlier launch)
        # EXPLICIT ASSUMPTION: Negative years means we launch earlier
        accelerated_years_to_launch = -time_saved_years
        accelerated_schedule = calculate_sales_schedule(accelerated_years_to_launch, peak_annual_sales, assumptions)
        accelerated_npv = sum(year['npv'] for year in accelerated_schedule)
        
        # NPV gain from acceleration
        # This is the PRIMARY value estimate (most accurate)
        npv_sales_gain = accelerated_npv - baseline_npv
        
        # Method 3: Patent life extension value
        # EXPLICIT ASSUMPTION: Each year saved = one additional year of exclusivity
        # This extends the revenue-generating period before generic competition
        if time_saved_years >= 1:
            years_extended = int(time_saved_years)
            patent_extension_value = 0
            base_patent_years = assumptions['patent_life_remaining_years']
            
            # ASSUMPTION: Extended years generate peak sales, discounted heavily
            for i in range(years_extended):
                year = base_patent_years + i
                discount_factor = 1 / ((1 + assumptions['discount_rate']) ** year)
                patent_extension_value += peak_annual_sales * discount_factor
        else:
            # Partial year extension (pro-rated)
            year = assumptions['patent_life_remaining_years']
            discount_factor = 1 / ((1 + assumptions['discount_rate']) ** year)
            patent_extension_value = peak_annual_sales * time_saved_years * discount_factor
        
        # Total value = NPV method (most comprehensive and accurate)
        total_value = npv_sales_gain
        
        # Compile all assumptions for transparency
        assumptions_output = {
            'Peak Annual Sales': f"${peak_annual_sales:,.0f}",
            'Daily Sales at Peak': f"${daily_sales_at_peak:,.0f}",
            'Daily Sales Calculation': f"Peak Annual Sales ÷ 365 days = ${peak_annual_sales:,.0f} ÷ 365 = ${daily_sales_at_peak:,.0f}/day",
            'Time Saved (Days)': f"{time_saved_days} days",
            'Time Saved (Years)': f"{time_saved_years:.2f} years",
            'Years to Peak Sales': f"{assumptions['time_to_peak_years']} years (industry standard)",
            'Peak Sales Duration': f"{assumptions['peak_sales_duration_years']} years before patent cliff",
            'Discount Rate (NPV)': f"{assumptions['discount_rate']*100:.0f}% (pharma industry standard)",
            'Market Category': assumptions['market_size_category'].title(),
            'Therapeutic Area': assumptions['therapeutic_area'].replace('_', ' ').title(),
            'Sales Ramp': f"Year 1: {assumptions['sales_year_1_pct']*100:.0f}%, Year 2: {assumptions['sales_year_2_pct']*100:.0f}%, Year 3: {assumptions['sales_year_3_pct']*100:.0f}%, Year 4: {assumptions['sales_year_4_pct']*100:.0f}%, Year 5: {assumptions['sales_year_5_pct']*100:.0f}% of peak",
            'Calculation Method': 'Net Present Value (NPV) with S-curve ramp-up',
            'Patent Life Assumption': f"{assumptions['patent_life_remaining_years']} years remaining at current stage",
            'NPV Formula': 'Sum of (Annual Sales × Discount Factor) for each year, where Discount Factor = 1/(1+r)^n',
            'Direct Sales Formula': f"Time Saved (days) × Daily Sales = {time_saved_days} days × ${daily_sales_at_peak:,.0f}/day"
        }
        
        return ProductSalesImpact(
            is_new_drug=True,
            peak_annual_sales=peak_annual_sales,
            daily_sales=daily_sales_at_peak,
            time_saved_days=time_saved_days,
            direct_sales_gain=direct_sales_gain,
            npv_sales_gain=npv_sales_gain,
            patent_life_extension_value=patent_extension_value,
            total_value=total_value,
            assumptions=assumptions_output,
            market_size_category=assumptions['market_size_category']
        )
    
    def _detect_indication(self) -> Dict[str, str]:
        """Detect therapeutic area and specific indication"""
        
        therapeutic_areas = {
            'oncology': ['cancer', 'tumor', 'tumour', 'oncology', 'carcinoma', 'lymphoma', 'leukemia', 
                        'leukaemia', 'melanoma', 'sarcoma', 'myeloma', 'glioblastoma', 'malignancy',
                        'neoplasm', 'metastatic', 'metastases'],
            'cardiology': ['cardiac', 'heart', 'cardiovascular', 'hypertension', 'heart failure', 
                          'arrhythmia', 'atrial fibrillation', 'coronary', 'myocardial'],
            'neurology': ['neurological', 'alzheimer', 'parkinson', 'multiple sclerosis', 'epilepsy', 
                         'stroke', 'migraine', 'neuropathy', 'dementia', 'als', 'huntington'],
            'respiratory': ['asthma', 'copd', 'pulmonary', 'respiratory', 'lung', 'bronchial'],
            'immunology': ['rheumatoid arthritis', 'lupus', 'psoriasis', 'crohn', 'ulcerative colitis', 
                          'autoimmune', 'inflammatory bowel'],
            'endocrinology': ['diabetes', 'thyroid', 'metabolic', 'obesity', 'endocrine'],
            'infectious_disease': ['hiv', 'hepatitis', 'infection', 'viral', 'bacterial', 'covid', 
                                   'influenza', 'pneumonia'],
            'rare_disease': ['orphan', 'rare disease', 'ultra-rare', 'ultra rare'],
            'psychiatry': ['depression', 'anxiety', 'schizophrenia', 'bipolar', 'psychiatric', 
                          'mental health', 'ptsd'],
        }
        
        detected_area = 'general'
        specific_indication = 'Not specified'
        
        for area, keywords in therapeutic_areas.items():
            for keyword in keywords:
                if keyword in self.text:
                    detected_area = area
                    specific_indication = keyword
                    break
            if detected_area != 'general':
                break
        
        is_rare = any(term in self.text for term in ['rare disease', 'orphan', 'ultra-rare', 'ultra rare'])
        
        return {
            'therapeutic_area': detected_area,
            'specific_indication': specific_indication,
            'is_rare': is_rare
        }
    
    def _estimate_population_size(self) -> str:
        """Estimate target population size"""
        sample_patterns = [
            r'(\d+)\s*(?:patients?|subjects?|participants?)',
            r'sample\s*size[:\s]*(\d+)',
            r'enroll(?:ment)?[:\s]*(\d+)',
            r'target[:\s]*(\d+)',
            r'approximately\s*(\d+)'
        ]
        
        sizes = []
        for pattern in sample_patterns:
            matches = re.findall(pattern, self.text)
            sizes.extend([int(m) for m in matches if m.isdigit() and int(m) > 10 and int(m) < 10000])
        
        if sizes:
            max_size = max(sizes)
            if max_size < 50:
                return 'small'
            elif max_size < 300:
                return 'medium'
            else:
                return 'large'
        
        # Default based on phase if not found
        if self.phase in ['Phase 1', 'Phase 1/2']:
            return 'small'
        elif self.phase in ['Phase 2', 'Phase 2/3']:
            return 'medium'
        elif self.phase == 'Phase 3':
            return 'large'
        
        return 'medium'
    
    def calculate_enrollment_acceleration(self, results: Dict[str, TrafficLightResult]) -> EnrollmentImpact:
        """Calculate potential enrollment acceleration based on improvements"""
        
        baseline_timelines = {
            'Phase 1': {'small': 6, 'medium': 9, 'large': 12},
            'Phase 1/2': {'small': 9, 'medium': 12, 'large': 15},
            'Phase 2': {'small': 12, 'medium': 18, 'large': 24},
            'Phase 2/3': {'small': 18, 'medium': 27, 'large': 36},
            'Phase 3': {'small': 24, 'medium': 36, 'large': 48},
            'Phase 4': {'small': 12, 'medium': 18, 'large': 24},
            'Phase Not Specified': {'small': 12, 'medium': 18, 'large': 24}
        }
        
        ta_multipliers = {
            'oncology': 1.0,
            'cardiology': 1.1,
            'neurology': 1.3,
            'respiratory': 1.1,
            'immunology': 1.2,
            'endocrinology': 1.0,
            'infectious_disease': 1.2,
            'rare_disease': 2.0,
            'psychiatry': 1.4,
            'general': 1.2
        }
        
        indication = self.indication
        therapeutic_area = indication['therapeutic_area']
        
        baseline = baseline_timelines.get(self.phase, baseline_timelines['Phase Not Specified'])[self.patient_population_size]
        baseline *= ta_multipliers.get(therapeutic_area, 1.2)
        
        if indication['is_rare']:
            baseline *= 1.5
        
        improvements = []
        total_improvement_percent = 0
        key_drivers = []
        
        patient_burden = results['patient_burden']
        if patient_burden.status == 'red':
            if patient_burden.score < 40:
                improvements.append(('High patient burden', 40))
                key_drivers.append("Reducing patient burden from RED to YELLOW: 25% faster enrollment")
                total_improvement_percent += 25
        elif patient_burden.status == 'yellow':
            improvements.append(('Moderate patient burden', 15))
            key_drivers.append("Reducing patient burden from YELLOW to GREEN: 15% faster enrollment")
            total_improvement_percent += 15
        
        dct = results['dct_suitability']
        if dct.status == 'green' or dct.score >= 60:
            improvements.append(('DCT implementation', 30))
            key_drivers.append("Implementing DCT/hybrid model: 30% faster enrollment")
            total_improvement_percent += 30
        elif dct.status == 'yellow':
            improvements.append(('Partial DCT implementation', 15))
            key_drivers.append("Implementing partial DCT elements: 15% faster enrollment")
            total_improvement_percent += 15
        
        contracting = results['contracting']
        if contracting.status == 'red':
            improvements.append(('Complex contracting delays', -20))
            key_drivers.append("Streamlining contracting (RED to YELLOW): 15% time reduction")
            total_improvement_percent += 15
        elif contracting.status == 'yellow':
            improvements.append(('Moderate contracting delays', -10))
            key_drivers.append("Optimizing contracting process: 10% time reduction")
            total_improvement_percent += 10
        
        if 'multinational' in self.text or 'multi-site' in self.text:
            if contracting.status != 'green':
                key_drivers.append("Using master agreements for multi-site: 10% faster activation")
                total_improvement_percent += 10
        
        tech_terms = ['epro', 'econsent', 'remote monitoring', 'wearable', 'telemedicine']
        tech_count = sum(1 for term in tech_terms if term in self.text)
        if tech_count >= 2:
            key_drivers.append(f"Technology enablement ({tech_count} technologies): 12% improvement")
            total_improvement_percent += 12
        
        if patient_burden.status == 'red' and len(patient_burden.factors) > 5:
            key_drivers.append("Protocol simplification: 20% improvement in retention")
            total_improvement_percent += 8
        
        if indication['is_rare']:
            if 'patient advocacy' not in self.text and 'registry' not in self.text:
                key_drivers.append("Engaging patient advocacy groups for rare disease: 25% improvement")
                total_improvement_percent += 25
        
        if self.phase == 'Phase 3' and self.patient_population_size == 'large':
            key_drivers.append("Data-driven site selection for large Phase 3: 15% improvement")
            total_improvement_percent += 15
        
        total_improvement_percent = min(total_improvement_percent, 60)
        
        improved_months = baseline * (1 - total_improvement_percent / 100)
        time_saved = baseline - improved_months
        
        cost_per_month = {
            'Phase 1': 100000,
            'Phase 1/2': 150000,
            'Phase 2': 200000,
            'Phase 2/3': 300000,
            'Phase 3': 400000,
            'Phase 4': 150000,
            'Phase Not Specified': 200000
        }
        
        monthly_cost = cost_per_month.get(self.phase, 200000)
        total_savings = time_saved * monthly_cost
        
        if total_savings >= 1000000:
            cost_savings_str = f"${total_savings/1000000:.1f}M - ${total_savings*1.5/1000000:.1f}M"
        else:
            cost_savings_str = f"${total_savings/1000:.0f}K - ${total_savings*1.5/1000:.0f}K"
        
        return EnrollmentImpact(
            baseline_months=round(baseline, 1),
            improved_months=round(improved_months, 1),
            time_saved_months=round(time_saved, 1),
            time_saved_percent=round(total_improvement_percent, 1),
            cost_savings_estimate=cost_savings_str,
            key_drivers=key_drivers if key_drivers else ["No significant improvements identified"]
        )
    
    def analyze_patient_burden(self) -> TrafficLightResult:
        score = 100
        factors = []
        recommendations = []
        
        visit_patterns = [
            r'(\d+)\s*(?:study\s*)?visits?',
            r'visit\s*(?:schedule|frequency)',
            r'(\d+)\s*(?:weeks?|months?)\s*(?:of\s*)?(?:treatment|follow[- ]?up)'
        ]
        
        visits = []
        for pattern in visit_patterns:
            matches = re.findall(pattern, self.text)
            visits.extend([int(m) for m in matches if m.isdigit()])
        
        if visits:
            max_visits = max(visits)
            if max_visits > 20:
                score -= 30
                factors.append(f"High visit frequency detected ({max_visits} visits)")
                recommendations.append("Consider remote visits or combining assessments")
            elif max_visits > 10:
                score -= 15
                factors.append(f"Moderate visit frequency ({max_visits} visits)")
        
        invasive_procedures = {
            'biopsy': 20,
            'bone marrow': 25,
            'lumbar puncture': 25,
            'endoscopy': 15,
            'bronchoscopy': 20,
            'colonoscopy': 20,
            'surgical': 30
        }
        
        for procedure, penalty in invasive_procedures.items():
            if procedure in self.text:
                score -= penalty
                factors.append(f"Invasive procedure: {procedure}")
                recommendations.append(f"Ensure clear patient communication about {procedure}")
        
        complex_terms = [
            'titration', 'dose escalation', 'multiple daily doses',
            'infusion', 'subcutaneous injection', 'intramuscular'
        ]
        
        for term in complex_terms:
            if term in self.text:
                score -= 5
                factors.append(f"Complex administration: {term}")
        
        if 'quality of life' in self.text or 'qol' in self.text or 'questionnaire' in self.text:
            qol_count = self.text.count('questionnaire') + self.text.count('diary')
            if qol_count > 5:
                score -= 10
                factors.append(f"Multiple questionnaires/diaries ({qol_count})")
                recommendations.append("Consider electronic PRO systems to reduce burden")
        
        score = max(0, min(100, score))
        if score >= 70:
            status = 'green'
        elif score >= 40:
            status = 'yellow'
            recommendations.append("Review protocol with patient advocacy groups")
        else:
            status = 'red'
            recommendations.append("Significant protocol simplification recommended")
            recommendations.append("Consider patient burden as a key retention risk")
        
        return TrafficLightResult(
            category="Patient Burden",
            status=status,
            score=score,
            factors=factors if factors else ["No significant burden factors detected"],
            recommendations=recommendations if recommendations else ["Protocol appears patient-friendly"]
        )
    
    def analyze_dct_suitability(self) -> TrafficLightResult:
        score = 50
        factors = []
        recommendations = []
        
        dct_positive = {
            'telemedicine': 15,
            'remote monitoring': 15,
            'wearable': 15,
            'home visit': 15,
            'econsent': 10,
            'epro': 10,
            'patient reported outcome': 10,
            'virtual visit': 15,
            'telehealth': 15,
            'mobile health': 10,
            'bring your own device': 10,
            'byod': 10
        }
        
        for term, points in dct_positive.items():
            if term in self.text:
                score += points
                factors.append(f"DCT-friendly: {term}")
        
        dct_negative = {
            'bone marrow': 30,
            'biopsy': 25,
            'imaging': 20,
            'infusion': 25,
            'iv administration': 25,
            'intravenous': 25,
            'radiation': 30,
            'surgical': 35,
            'hospitalization': 30,
            'intensive care': 35
        }
        
        for term, penalty in dct_negative.items():
            if term in self.text:
                score -= penalty
                factors.append(f"DCT challenge: {term}")
        
        if 'oral' in self.text and ('tablet' in self.text or 'capsule' in self.text):
            score += 20
            factors.append("Oral medication (DCT-friendly)")
            recommendations.append("Consider smart pill bottles for adherence monitoring")
        
        rare_indicators = ['orphan', 'rare disease', 'prevalence']
        if any(term in self.text for term in rare_indicators):
            score -= 10
            factors.append("Rare disease may limit DCT implementation")
        
        score = max(0, min(100, score))
        if score >= 70:
            status = 'green'
            recommendations.append("Excellent candidate for full DCT implementation")
            recommendations.append("Consider hybrid model with home health nurses")
        elif score >= 40:
            status = 'yellow'
            recommendations.append("Hybrid DCT model recommended")
            recommendations.append("Implement remote visits where possible")
        else:
            status = 'red'
            recommendations.append("Traditional site-based model more appropriate")
            recommendations.append("Consider DCT elements for follow-up visits only")
        
        return TrafficLightResult(
            category="DCT Suitability",
            status=status,
            score=score,
            factors=factors if factors else ["Limited DCT indicators found"],
            recommendations=recommendations
        )
    
    def analyze_study_budget(self) -> TrafficLightResult:
        score = 100
        factors = []
        recommendations = []
        
        expensive_procedures = {
            'genomic': 25,
            'biomarker': 20,
            'imaging': 15,
            'mri': 20,
            'pet scan': 25,
            'ct scan': 15,
            'genetic testing': 25,
            'sequencing': 30,
            'flow cytometry': 20,
            'mass spectrometry': 25
        }
        
        for procedure, penalty in expensive_procedures.items():
            if procedure in self.text:
                score -= penalty
                factors.append(f"High-cost procedure: {procedure}")
        
        if 'rare disease' in self.text or 'orphan' in self.text:
            score -= 20
            factors.append("Rare disease = higher recruitment costs")
            recommendations.append("Budget for patient travel and lodging")
        
        visit_numbers = re.findall(r'(\d+)\s*visits?', self.text)
        if visit_numbers:
            max_visits = max([int(v) for v in visit_numbers])
            if max_visits > 15:
                score -= 20
                factors.append(f"High visit frequency ({max_visits}) increases costs")
                recommendations.append("Negotiate per-visit payments carefully")
        
        duration_patterns = [
            r'(\d+)\s*(?:year|yr)',
            r'(\d+)\s*(?:month|mo)'
        ]
        
        for pattern in duration_patterns:
            matches = re.findall(pattern, self.text)
            if matches:
                if 'year' in pattern and matches:
                    years = max([int(m) for m in matches])
                    if years > 2:
                        score -= 15
                        factors.append(f"Long study duration ({years} years)")
                        recommendations.append("Plan for inflation and retention costs")
        
        if 'central lab' in self.text or 'central laboratory' in self.text:
            score -= 10
            factors.append("Central laboratory costs")
        
        if 'adjudication' in self.text or 'committee' in self.text:
            score -= 10
            factors.append("Adjudication committee costs")
        
        if self.phase in ['Phase 3', 'Phase 2/3']:
            score -= 15
            factors.append(f"{self.phase} = larger scale and higher costs")
        elif self.phase == 'Phase 1':
            factors.append("Phase 1 = smaller scale")
        
        score = max(0, min(100, score))
        if score >= 70:
            status = 'green'
            recommendations.append("Budget appears manageable")
        elif score >= 40:
            status = 'yellow'
            recommendations.append("Moderate budget complexity - detailed cost analysis needed")
            recommendations.append("Consider competitive bidding for specialized services")
        else:
            status = 'red'
            recommendations.append("High budget complexity - extensive financial planning required")
            recommendations.append("Engage budget specialists early")
            recommendations.append("Consider staged funding approach")
        
        return TrafficLightResult(
            category="Study Budget",
            status=status,
            score=score,
            factors=factors if factors else ["Standard budget complexity"],
            recommendations=recommendations
        )
    
    def analyze_contracting_complexity(self) -> TrafficLightResult:
        score = 100
        factors = []
        recommendations = []
        
        country_matches = re.findall(r'(\d+)\s*(?:countries|sites)', self.text)
        if country_matches:
            max_count = max([int(m) for m in country_matches])
            if max_count > 50:
                score -= 30
                factors.append(f"Large number of sites/countries ({max_count})")
                recommendations.append("Use master agreement templates")
                recommendations.append("Consider central IRB/EC where possible")
            elif max_count > 20:
                score -= 15
                factors.append(f"Moderate number of sites ({max_count})")
        
        if 'multinational' in self.text or 'international' in self.text:
            score -= 15
            factors.append("Multinational study increases contracting complexity")
            recommendations.append("Engage local legal experts in each country")
        
        if 'biologic' in self.text or 'cell therapy' in self.text or 'gene therapy' in self.text:
            score -= 20
            factors.append("Advanced therapy = complex IP and material transfer agreements")
            recommendations.append("Early IP and MTA negotiations critical")
        
        if 'academic' in self.text or 'university' in self.text:
            score -= 15
            factors.append("Academic sites = longer negotiation timelines")
            recommendations.append("Allow 3-6 months for academic site contracts")
        
        special_pops = ['pediatric', 'paediatric', 'children', 'pregnant', 'prisoner']
        for pop in special_pops:
            if pop in self.text:
                score -= 10
                factors.append(f"Special population ({pop}) = additional oversight")
                recommendations.append(f"Additional ethical and legal review for {pop} population")
        
        if 'device' in self.text or 'combination product' in self.text:
            score -= 10
            factors.append("Device/combination product = additional agreements")
        
        if 'data sharing' in self.text or 'data transparency' in self.text:
            score -= 10
            factors.append("Data sharing requirements increase contract complexity")
        
        score = max(0, min(100, score))
        if score >= 70:
            status = 'green'
            recommendations.append("Standard contracting process should suffice")
        elif score >= 40:
            status = 'yellow'
            recommendations.append("Moderate contracting complexity - allow extra time")
            recommendations.append("Consider staged site activation")
        else:
            status = 'red'
            recommendations.append("High contracting complexity - significant delays likely")
            recommendations.append("Engage legal and contracting teams immediately")
            recommendations.append("Build 6+ months into timeline for contracting")
        
        return TrafficLightResult(
            category="Site Contracting Complexity",
            status=status,
            score=score,
            factors=factors if factors else ["Standard contracting complexity"],
            recommendations=recommendations
        )

def extract_text_from_pdf(file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file) -> str:
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_txt(file) -> str:
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

def get_traffic_light_emoji(status: str) -> str:
    if status == 'green':
        return "🟢"
    elif status == 'yellow':
        return "🟡"
    else:
        return "🔴"

def get_status_color(status: str) -> str:
    if status == 'green':
        return "#28a745"
    elif status == 'yellow':
        return "#ffc107"
    else:
        return "#dc3545"

def format_currency(value: float) -> str:
    """Format large currency values"""
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:.0f}"

def main():
    st.set_page_config(
        page_title="Clinical Trial Protocol Analyzer",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Clinical Trial Protocol Analyzer with Revenue Impact")
    st.markdown("""
    Upload your clinical trial protocol to receive automated analysis including **enrollment acceleration** 
    and **product sales forecasting** for new drug applications.
    """)
    
    st.sidebar.header("Upload Protocol")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Upload your clinical trial protocol in PDF, DOCX, or TXT format"
    )
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        with st.spinner("Processing document..."):
            if file_type == 'pdf':
                text = extract_text_from_pdf(uploaded_file)
            elif file_type == 'docx':
                text = extract_text_from_docx(uploaded_file)
            elif file_type == 'txt':
                text = extract_text_from_txt(uploaded_file)
            else:
                st.error("Unsupported file type")
                return
        
        if text:
            st.success(f"✅ Document processed successfully ({len(text.split())} words)")
            
            analyzer = ProtocolAnalyzer(text)
            
            # Display study characteristics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                phase_color = "#007bff" if analyzer.phase != "Phase Not Specified" else "#6c757d"
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 8px; background-color: {phase_color}20; border: 2px solid {phase_color}">
                    <h4 style="margin: 0; color: {phase_color}">Study Phase</h4>
                    <h2 style="margin: 5px 0; color: {phase_color}">{analyzer.phase}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                indication_display = analyzer.indication['specific_indication'].title()
                st.metric("Indication", indication_display)
            with col3:
                ta_display = analyzer.indication['therapeutic_area'].replace('_', ' ').title()
                st.metric("Therapeutic Area", ta_display)
            with col4:
                drug_status = "New Drug" if analyzer.is_new_drug else "Other"
                drug_color = "#28a745" if analyzer.is_new_drug else "#6c757d"
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 8px; background-color: {drug_color}20; border: 2px solid {drug_color}">
                    <h4 style="margin: 0; color: {drug_color}">Study Type</h4>
                    <h2 style="margin: 5px 0; color: {drug_color}">{drug_status}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner("Analyzing protocol..."):
                results = {
                    "patient_burden": analyzer.analyze_patient_burden(),
                    "dct_suitability": analyzer.analyze_dct_suitability(),
                    "budget": analyzer.analyze_study_budget(),
                    "contracting": analyzer.analyze_contracting_complexity()
                }
                
                enrollment_impact = analyzer.calculate_enrollment_acceleration(results)
                sales_impact = analyzer.calculate_product_sales_impact(enrollment_impact)
            
            # PRODUCT SALES IMPACT SECTION
            if sales_impact.is_new_drug:
                st.markdown("---")
                st.header("💰 Product Sales Impact from Accelerated Timeline")
                
                st.info("🔔 **New Drug Detected** - Additional revenue analysis included based on earlier market entry")
                
                value_cols = st.columns(3)
                
                with value_cols[0]:
                    st.markdown(f"""
                    <div style="padding: 25px; border-radius: 10px; background-color: #28a74530; border: 3px solid #28a745">
                        <h3 style="margin: 0; color: #28a745">💵 Total NPV Gain</h3>
                        <h1 style="margin: 10px 0; color: #28a745; font-size: 2.5rem">{format_currency(sales_impact.total_value)}</h1>
                        <p style="margin: 0; font-weight: bold">Net Present Value of earlier launch</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with value_cols[1]:
                    st.markdown(f"""
                    <div style="padding: 25px; border-radius: 10px; background-color: #007bff30; border: 3px solid #007bff">
                        <h3 style="margin: 0; color: #007bff">📈 Peak Daily Sales</h3>
                        <h1 style="margin: 10px 0; color: #007bff; font-size: 2.5rem">{format_currency(sales_impact.daily_sales)}</h1>
                        <p style="margin: 0; font-weight: bold">Revenue per day at peak</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with value_cols[2]:
                    st.markdown(f"""
                    <div style="padding: 25px; border-radius: 10px; background-color: #ffc10730; border: 3px solid #ffc107">
                        <h3 style="margin: 0; color: #856404">⏱️ Direct Revenue Gain</h3>
                        <h1 style="margin: 10px 0; color: #856404; font-size: 2.5rem">{format_currency(sales_impact.direct_sales_gain)}</h1>
                        <p style="margin: 0; font-weight: bold">Simple: Days saved × Daily sales</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detailed assumptions section
                st.subheader("📊 Revenue Model Assumptions & Methodology")
                
                assumptions_col1, assumptions_col2 = st.columns(2)
                
                with assumptions_col1:
                    st.markdown("### 💼 Market & Sales Assumptions")
                    
                    assumptions_table = []
                    key_assumptions = [
                        'Peak Annual Sales',
                        'Daily Sales at Peak',
                        'Daily Sales Calculation',
                        'Time Saved (Days)',
                        'Time Saved (Years)',
                        'Market Category',
                        'Therapeutic Area'
                    ]
                    
                    for key in key_assumptions:
                        if key in sales_impact.assumptions:
                            assumptions_table.append({
                                'Parameter': key,
                                'Value': sales_impact.assumptions[key]
                            })
                    
                    st.table(pd.DataFrame(assumptions_table))
                
                with assumptions_col2:
                    st.markdown("### 📈 Financial Model Parameters")
                    
                    financial_table = []
                    financial_keys = [
                        'Years to Peak Sales',
                        'Peak Sales Duration',
                        'Discount Rate (NPV)',
                        'Patent Life Assumption',
                        'Calculation Method'
                    ]
                    
                    for key in financial_keys:
                        if key in sales_impact.assumptions:
                            financial_table.append({
                                'Parameter': key,
                                'Value': sales_impact.assumptions[key]
                            })
                    
                    st.table(pd.DataFrame(financial_table))
                
                # Sales ramp explanation
                st.markdown("### 📊 Sales Ramp-Up Profile")
                st.info(sales_impact.assumptions.get('Sales Ramp', 'Standard S-curve ramp'))
                
                # Calculation formulas
                st.markdown("### 🧮 Calculation Formulas")
                col1, col2 = st.columns(2)
                with col1:
                    st.code(sales_impact.assumptions.get('Daily Sales Calculation', ''), language='text')
                with col2:
                    st.code(sales_impact.assumptions.get('Direct Sales Formula', ''), language='text')
                
                # Create visualization
                st.markdown("### 📉 Revenue Impact Visualization")
                
                years = list(range(1, 16))
                baseline_revenues = []
                accelerated_revenues = []
                
                ramp_percentages = [0.15, 0.35, 0.60, 0.85, 1.0]
                peak_sales = sales_impact.peak_annual_sales
                time_saved_years = enrollment_impact.time_saved_months / 12
                
                for year in years:
                    if year <= 5:
                        baseline_rev = peak_sales * ramp_percentages[year-1]
                    elif year <= 10:
                        baseline_rev = peak_sales
                    elif year <= 13:
                        baseline_rev = peak_sales * (0.7 - (year-11) * 0.2)
                    else:
                        baseline_rev = 0
                    baseline_revenues.append(baseline_rev)
                    
                    accel_year = year - time_saved_years
                    if accel_year <= 0:
                        accelerated_rev = 0
                    elif accel_year <= 5:
                        idx = max(0, min(4, int(accel_year) - 1))
                        accelerated_rev = peak_sales * ramp_percentages[idx]
                    elif accel_year <= 10:
                        accelerated_rev = peak_sales
                    elif accel_year <= 13:
                        accelerated_rev = peak_sales * (0.7 - (accel_year-11) * 0.2)
                    else:
                        accelerated_rev = 0
                    accelerated_revenues.append(accelerated_rev)
                
                revenue_df = pd.DataFrame({
                    'Year': years,
                    'Baseline Revenue': baseline_revenues,
                    'Accelerated Revenue': accelerated_revenues
                })
                
                revenue_df['Baseline Cumulative'] = revenue_df['Baseline Revenue'].cumsum()
                revenue_df['Accelerated Cumulative'] = revenue_df['Accelerated Revenue'].cumsum()
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=revenue_df['Year'],
                    y=revenue_df['Baseline Revenue'],
                    name='Baseline Annual Revenue',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=revenue_df['Year'],
                    y=revenue_df['Accelerated Revenue'],
                    name='Accelerated Annual Revenue',
                    line=dict(color='green', width=3)
                ))
                
                fig.update_layout(
                    title='Annual Revenue Comparison: Baseline vs. Accelerated Timeline',
                    xaxis_title='Years from Today',
                    yaxis_title='Annual Revenue ($)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=revenue_df['Year'],
                    y=revenue_df['Baseline Cumulative'],
                    name='Baseline Cumulative',
                    fill='tonexty',
                    line=dict(color='red', width=2)
                ))
                
                fig2.add_trace(go.Scatter(
                    x=revenue_df['Year'],
                    y=revenue_df['Accelerated Cumulative'],
                    name='Accelerated Cumulative',
                    fill='tonexty',
                    line=dict(color='green', width=2)
                ))
                
                fig2.update_layout(
                    title='Cumulative Revenue Comparison',
                    xaxis_title='Years from Today',
                    yaxis_title='Cumulative Revenue ($)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                with st.expander("🔍 Detailed Calculation Methodology"):
                    st.markdown("""
                    ### Revenue Impact Calculation Methods
                    
                    We use **three complementary methods** to estimate the financial impact of accelerated timelines:
                    
                    #### 1. Direct Sales Gain (Conservative)
                    **Formula:** `Time Saved (days) × Daily Sales at Peak`
                    
                    - **Result:** {direct}
                    - **Daily Sales Calculation:** Peak Annual Sales ÷ 365 days
                    - **Assumption:** Simple multiplication - assumes immediate peak sales
                    - **Limitation:** Doesn't account for time value of money or sales ramp-up
                    
                    #### 2. Net Present Value (NPV) Method (Primary - Most Accurate)
                    **Formula:** `NPV(Accelerated Schedule) - NPV(Baseline Schedule)`
                    
                    - **Result:** {npv}
                    - **Methodology:**
                        - Projects year-by-year revenues for both scenarios
                        - Applies S-curve ramp-up (15% → 35% → 60% → 85% → 100% over 5 years)
                        - Discounts future cash flows at 10% annually
                        - Accounts for patent cliff after peak years
                    - **This is our PRIMARY estimate** (most comprehensive)
                    
                    #### 3. Patent Life Extension Value
                    **Formula:** `Additional Years × Peak Sales × Discount Factor`
                    
                    - **Result:** {patent}
                    - **Rationale:** Earlier launch = more market exclusivity before patent expiry
                    
                    ### Key Assumptions Explained
                    
                    - **Peak Sales Benchmarks** (based on therapeutic area):
                        - Oncology blockbusters: $5B (e.g., Keytruda)
                        - Immunology blockbusters: $6B (e.g., Humira)
                        - Rare disease: $250M - $2B with orphan premium
                    
                    - **Daily Sales:** Peak Annual Sales ÷ 365 days
                        - Example: ${peak_annual} ÷ 365 = **${daily}/day**
                    
                    - **Time to Peak:** 5 years is pharma industry standard
                    
                    - **Discount Rate:** 10% reflects pharma industry cost of capital
                    
                    - **S-Curve Ramp:** Reflects real-world adoption:
                        - Year 1: 15% (slow initial uptake)
                        - Year 2-4: Rapid acceleration
                        - Year 5: 100% (market saturation)
                    
                    ### Why NPV is Most Accurate
                    
                    ✅ Accounts for **time value of money**
                    ✅ Reflects **realistic sales ramp-up**
                    ✅ Includes **patent expiry effects**
                    ✅ Standard method in pharma financial planning
                    """.format(
                        direct=format_currency(sales_impact.direct_sales_gain),
                        npv=format_currency(sales_impact.npv_sales_gain),
                        patent=format_currency(sales_impact.patent_life_extension_value),
                        peak_annual=format_currency(sales_impact.peak_annual_sales),
                        daily=format_currency(sales_impact.daily_sales)
                    ))
                
                st.markdown("---")
            
            # ENROLLMENT ACCELERATION SECTION
            st.header("🚀 Enrollment Acceleration Opportunity")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #ff4b4b20; border: 2px solid #ff4b4b">
                    <h4 style="margin: 0; color: #ff4b4b">Baseline Timeline</h4>
                    <h1 style="margin: 10px 0; color: #ff4b4b">{enrollment_impact.baseline_months}</h1>
                    <p style="margin: 0; font-weight: bold">months</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #28a74520; border: 2px solid #28a745">
                    <h4 style="margin: 0; color: #28a745">Optimized Timeline</h4>
                    <h1 style="margin: 10px 0; color: #28a745">{enrollment_impact.improved_months}</h1>
                    <p style="margin: 0; font-weight: bold">months</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #007bff20; border: 2px solid #007bff">
                    <h4 style="margin: 0; color: #007bff">Time Saved</h4>
                    <h1 style="margin: 10px 0; color: #007bff">{enrollment_impact.time_saved_months}</h1>
                    <p style="margin: 0; font-weight: bold">months ({enrollment_impact.time_saved_percent}%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: #ffc10720; border: 2px solid #ffc107">
                    <h4 style="margin: 0; color: #856404">Trial Cost Savings</h4>
                    <h1 style="margin: 10px 0; color: #856404; font-size: 1.8rem">{enrollment_impact.cost_savings_estimate}</h1>
                    <p style="margin: 0; font-weight: bold">estimated</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("📊 Key Improvement Drivers")
            
            for idx, driver in enumerate(enrollment_impact.key_drivers, 1):
                st.markdown(f"**{idx}.** {driver}")
            
            st.markdown("---")
            
            # Protocol Analysis Dashboard
            st.header("📊 Protocol Analysis Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            columns = [col1, col2, col3, col4]
            result_keys = ["patient_burden", "dct_suitability", "budget", "contracting"]
            
            for col, key in zip(columns, result_keys):
                result = results[key]
                with col:
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: {get_status_color(result.status)}20; border: 2px solid {get_status_color(result.status)}">
                        <h3 style="margin: 0; color: {get_status_color(result.status)}">{get_traffic_light_emoji(result.status)} {result.category}</h3>
                        <h1 style="margin: 10px 0; color: {get_status_color(result.status)}">{result.score}/100</h1>
                        <p style="margin: 0; font-weight: bold; color: {get_status_color(result.status)}">{result.status.upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed results tabs
            st.header("📋 Detailed Analysis")
            
            tabs = st.tabs([r.category for r in results.values()])
            
            for tab, result in zip(tabs, results.values()):
                with tab:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Score", f"{result.score}/100")
                        st.metric("Status", result.status.upper())
                    
                    with col2:
                        st.progress(result.score / 100)
                    
                    st.subheader("🔍 Key Factors Identified")
                    for factor in result.factors:
                        st.markdown(f"- {factor}")
                    
                    st.subheader("💡 Recommendations")
                    for rec in result.recommendations:
                        st.markdown(f"- {rec}")
            
            # Export functionality
            st.markdown("---")
            st.header("📥 Export Report")
            
            report = f"""
CLINICAL TRIAL PROTOCOL ANALYSIS REPORT
{'='*60}

Document: {uploaded_file.name}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

STUDY CHARACTERISTICS
{'='*60}
Phase: {analyzer.phase}
Therapeutic Area: {analyzer.indication['therapeutic_area'].replace('_', ' ').title()}
Indication: {analyzer.indication['specific_indication'].title()}
Estimated Population Size: {analyzer.patient_population_size.title()}
Study Type: {'New Drug Application' if analyzer.is_new_drug else 'Other'}

"""
            
            if sales_impact.is_new_drug:
                report += f"""
PRODUCT SALES IMPACT FROM ACCELERATED TIMELINE
{'='*60}
Total NPV Gain: {format_currency(sales_impact.total_value)}
Direct Revenue Gain: {format_currency(sales_impact.direct_sales_gain)}
Patent Life Extension Value: {format_currency(sales_impact.patent_life_extension_value)}

Market Assumptions:
- Peak Annual Sales: {sales_impact.assumptions['Peak Annual Sales']}
- Daily Sales at Peak: {sales_impact.assumptions['Daily Sales at Peak']}
- Daily Sales Calculation: {sales_impact.assumptions['Daily Sales Calculation']}
- Time Saved: {sales_impact.assumptions['Time Saved (Days)']} ({sales_impact.assumptions['Time Saved (Years)']})
- Market Category: {sales_impact.assumptions['Market Category']}
- Therapeutic Area: {sales_impact.assumptions['Therapeutic Area']}

Financial Model Parameters:
- Years to Peak Sales: {sales_impact.assumptions['Years to Peak Sales']}
- Peak Sales Duration: {sales_impact.assumptions['Peak Sales Duration']}
- Discount Rate: {sales_impact.assumptions['Discount Rate (NPV)']}
- Patent Life Assumption: {sales_impact.assumptions['Patent Life Assumption']}
- Calculation Method: {sales_impact.assumptions['Calculation Method']}
- Sales Ramp Profile: {sales_impact.assumptions['Sales Ramp']}

Formulas Used:
- {sales_impact.assumptions.get('Daily Sales Calculation', '')}
- {sales_impact.assumptions.get('Direct Sales Formula', '')}

"""
            
            report += f"""
ENROLLMENT ACCELERATION OPPORTUNITY
{'='*60}
Baseline Enrollment Timeline: {enrollment_impact.baseline_months} months
Optimized Enrollment Timeline: {enrollment_impact.improved_months} months
Time Saved: {enrollment_impact.time_saved_months} months ({enrollment_impact.time_saved_percent}%)
Trial Cost Savings: {enrollment_impact.cost_savings_estimate}

Key Improvement Drivers:
"""
            for idx, driver in enumerate(enrollment_impact.key_drivers, 1):
                report += f"{idx}. {driver}\n"
            
            report += f"""

PROTOCOL ANALYSIS SUMMARY
{'='*60}
"""
            for result in results.values():
                report += f"""
{result.category}
Status: {result.status.upper()} ({result.score}/100)

Key Factors:
"""
                for factor in result.factors:
                    report += f"  - {factor}\n"
                
                report += "\nRecommendations:\n"
                for rec in result.recommendations:
                    report += f"  - {rec}\n"
                report += "\n" + "-"*60 + "\n"
            
            # Create comprehensive CSV
            summary_data = {
                'Metric': [
                    'Study Phase',
                    'Therapeutic Area',
                    'Study Type',
                    'Baseline Enrollment (months)',
                    'Optimized Enrollment (months)',
                    'Time Saved (months)',
                    'Time Saved (%)',
                    'Trial Cost Savings',
                    'Patient Burden Score',
                    'DCT Suitability Score',
                    'Budget Complexity Score',
                    'Contracting Complexity Score'
                ],
                'Value': [
                    analyzer.phase,
                    analyzer.indication['therapeutic_area'].replace('_', ' ').title(),
                    'New Drug' if analyzer.is_new_drug else 'Other',
                    enrollment_impact.baseline_months,
                    enrollment_impact.improved_months,
                    enrollment_impact.time_saved_months,
                    f"{enrollment_impact.time_saved_percent}%",
                    enrollment_impact.cost_savings_estimate,
                    results['patient_burden'].score,
                    results['dct_suitability'].score,
                    results['budget'].score,
                    results['contracting'].score
                ]
            }
            
            if sales_impact.is_new_drug:
                summary_data['Metric'].extend([
                    'Total NPV Gain',
                    'Direct Revenue Gain',
                    'Peak Annual Sales',
                    'Daily Sales at Peak',
                    'Market Category'
                ])
                summary_data['Value'].extend([
                    format_currency(sales_impact.total_value),
                    format_currency(sales_impact.direct_sales_gain),
                    format_currency(sales_impact.peak_annual_sales),
                    format_currency(sales_impact.daily_sales),
                    sales_impact.market_size_category.title()
                ])
            
            summary_df = pd.DataFrame(summary_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download Full Report (TXT)",
                    data=report,
                    file_name=f"protocol_analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Summary (CSV)",
                    data=csv,
                    file_name=f"protocol_summary_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.error("Could not extract text from the document. Please check the file and try again.")
    
    else:
        st.info("👈 Please upload a clinical trial protocol document using the sidebar to begin analysis. You can download an example here: https://cdn.clinicaltrials.gov/large-docs/52/NCT03235752/Prot_000.pdf")
        
        st.markdown("""
        ### 🎯 What You'll Get:
        
        1. **Accurate Phase Detection** - Recognizes Roman numerals (Phase I, II, III, IV) and combinations (Phase I/II)
        2. **Product Revenue Impact** (For New Drugs) - Estimated NPV gain from accelerated market entry
        3. **Enrollment Acceleration Forecast** - Time and cost savings from protocol optimization
        4. **Traffic Light Indicators** - Quick assessment of four critical areas
        
        ### 📊 Phase Detection Examples:
        
        The system recognizes various formats:
        - ✅ "Phase III study"
        - ✅ "Phase 3 clinical trial"
        - ✅ "Phase-II trial"
        - ✅ "Phase I/II combination"
        - ✅ "Phase 2/3 study"
        - ✅ Context clues: "first-in-human" → Phase 1, "pivotal" → Phase 3
        
        ### 💰 Revenue Calculations (New Drugs):
        
        **Explicit Daily Sales Formula:**
        `Daily Sales = Peak Annual Sales ÷ 365 days`
        
        **Example:** $1.5B peak annual sales = **$4,109,589 per day**
        
        All assumptions are transparent and based on industry benchmarks.
        """)

if __name__ == "__main__":
    main()