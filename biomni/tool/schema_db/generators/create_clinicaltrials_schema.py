import pickle

# ClinicalTrials.gov API v2 schema
clinicaltrials_schema = {
    'base_url': 'https://clinicaltrials.gov/api/v2',
    'description': 'ClinicalTrials.gov API v2 for accessing clinical trial data',
    'docs_url': 'https://clinicaltrials.gov/data-api/api',
    'license': 'Public domain (US government)',
    'api_version': '2.0.3',
    'format': 'JSON',
    'rate_limit': 'No explicit rate limit mentioned',
    
    'endpoints': {
        'version': {
            'description': 'Get API version information',
            'url': '/version',
            'method': 'GET'
        },
        'studies': {
            'description': 'Search and retrieve clinical studies',
            'url': '/studies',
            'method': 'GET',
            'parameters': {
                'query.cond': 'Search by condition/disease',
                'query.term': 'Search by general terms',
                'query.titles': 'Search in study titles',
                'query.intr': 'Search by intervention',
                'query.outc': 'Search by outcome measures',
                'query.spons': 'Search by sponsor',
                'query.lead': 'Search by lead sponsor',
                'query.id': 'Search by study ID',
                'query.patient': 'Search by patient characteristics',
                'query.locn': 'Search by location',
                'filter.overallStatus': 'Filter by study status',
                'filter.phase': 'Filter by study phase',
                'filter.studyType': 'Filter by study type',
                'filter.results': 'Filter by results availability',
                'filter.advanced': 'Advanced search filters',
                'pageSize': 'Number of results per page (max 1000)',
                'pageToken': 'Token for pagination',
                'countTotal': 'Include total count in response',
                'format': 'Response format (json, csv)',
                'fields': 'Specific fields to include in response'
            }
        },
        'studies_by_id': {
            'description': 'Get specific study by NCT ID',
            'url': '/studies/{nctId}',
            'method': 'GET',
            'parameters': {
                'format': 'Response format (json, csv)',
                'fields': 'Specific fields to include in response'
            }
        },
        'stats': {
            'description': 'Get statistics about studies',
            'url': '/stats',
            'method': 'GET',
            'parameters': {
                'query.cond': 'Condition for statistics',
                'query.term': 'Terms for statistics',
                'filter.overallStatus': 'Status filter for statistics'
            }
        }
    },
    
    'study_statuses': [
        'ACTIVE_NOT_RECRUITING',
        'COMPLETED',
        'ENROLLING_BY_INVITATION',
        'NOT_YET_RECRUITING',
        'RECRUITING',
        'SUSPENDED',
        'TERMINATED',
        'WITHDRAWN',
        'UNKNOWN'
    ],
    
    'study_phases': [
        'EARLY_PHASE1',
        'PHASE1',
        'PHASE1_PHASE2',
        'PHASE2',
        'PHASE2_PHASE3',
        'PHASE3',
        'PHASE4',
        'NA'
    ],
    
    'study_types': [
        'INTERVENTIONAL',
        'OBSERVATIONAL',
        'EXPANDED_ACCESS'
    ],
    
    'common_fields': [
        'protocolSection.identificationModule.nctId',
        'protocolSection.identificationModule.briefTitle',
        'protocolSection.identificationModule.officialTitle',
        'protocolSection.statusModule.overallStatus',
        'protocolSection.statusModule.startDateStruct',
        'protocolSection.statusModule.completionDateStruct',
        'protocolSection.sponsorCollaboratorsModule.leadSponsor',
        'protocolSection.conditionsModule.conditions',
        'protocolSection.designModule.studyType',
        'protocolSection.designModule.phases',
        'protocolSection.armsInterventionsModule.interventions',
        'protocolSection.outcomesModule.primaryOutcomes',
        'protocolSection.eligibilityModule.eligibilityCriteria',
        'protocolSection.contactsLocationsModule.locations'
    ],
    
    'examples': {
        'search_cancer': {
            'endpoint': '/studies',
            'parameters': {'query.cond': 'cancer', 'pageSize': 10}
        },
        'search_covid': {
            'endpoint': '/studies',
            'parameters': {'query.cond': 'COVID-19', 'filter.overallStatus': 'RECRUITING'}
        },
        'get_specific_study': {
            'endpoint': '/studies/NCT05013879',
            'parameters': {}
        },
        'search_by_intervention': {
            'endpoint': '/studies',
            'parameters': {'query.intr': 'aspirin', 'pageSize': 20}
        },
        'recruiting_phase3': {
            'endpoint': '/studies',
            'parameters': {
                'filter.overallStatus': 'RECRUITING',
                'filter.phase': 'PHASE3',
                'pageSize': 50
            }
        }
    },
    
    'pagination': {
        'description': 'API uses token-based pagination',
        'max_page_size': 1000,
        'default_page_size': 10,
        'next_page_token_field': 'nextPageToken'
    },
    
    'data_format': {
        'primary': 'JSON',
        'alternative': 'CSV',
        'date_format': 'ISO 8601',
        'text_format': 'CommonMark Markdown for rich text'
    }
}

with open('clinicaltrials.pkl', 'wb') as f:
    pickle.dump(clinicaltrials_schema, f)

print('ClinicalTrials.gov schema created successfully')
