from fastmcp import FastMCP
from tooluniverse.execute_function import ToolUniverse

mcp = FastMCP("ToolUniverse MCP", stateless_http=True)
engine = ToolUniverse()
engine.load_tools()


@mcp.tool()
def FDA_get_active_ingredient_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_active_ingredient_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_dosage_and_storage_information_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_dosage_and_storage_information_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_abuse_info(abuse_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_abuse_info",
            "arguments": {"abuse_info": abuse_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_abuse_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_abuse_info_by_drug_name", "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip}}
    )


@mcp.tool()
def FDA_get_drug_names_by_accessories(accessory_name: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_accessories",
            "arguments": {"accessory_name": accessory_name, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_accessories_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_accessories_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_active_ingredient(active_ingredient: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_active_ingredient",
            "arguments": {
                "active_ingredient": active_ingredient,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_manufacturer_name_NDC_number_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_manufacturer_name_NDC_number_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_application_number_NDC_number(
    application_manufacturer_or_NDC_info: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_application_number_NDC_number",
            "arguments": {
                "application_manufacturer_or_NDC_info": application_manufacturer_or_NDC_info,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_adverse_reaction(adverse_reaction: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_adverse_reaction",
            "arguments": {"adverse_reaction": adverse_reaction, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_adverse_reactions_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_adverse_reactions_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_alarm(alarm_type: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_alarm",
            "arguments": {"alarm_type": alarm_type, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_alarms_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_alarms_by_drug_name", "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip}}
    )


@mcp.tool()
def FDA_get_drug_names_by_animal_pharmacology_info(
    pharmacology_info: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_animal_pharmacology_info",
            "arguments": {
                "pharmacology_info": pharmacology_info,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_animal_pharmacology_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_animal_pharmacology_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_info_on_conditions_for_doctor_consultation(condition: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_info_on_conditions_for_doctor_consultation",
            "arguments": {"condition": condition, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_info_on_conditions_for_doctor_consultation_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_info_on_conditions_for_doctor_consultation_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_consulting_doctor_pharmacist_info(
    interaction_info: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_consulting_doctor_pharmacist_info",
            "arguments": {"interaction_info": interaction_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_info_on_consulting_doctor_pharmacist_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_info_on_consulting_doctor_pharmacist_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_assembly_installation_info(field_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_assembly_installation_info",
            "arguments": {"field_info": field_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_assembly_installation_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_assembly_installation_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_boxed_warning(warning_text: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_boxed_warning",
            "arguments": {"warning_text": warning_text, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_boxed_warning_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_boxed_warning_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_calibration_instructions(calibration_instructions: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_calibration_instructions",
            "arguments": {"calibration_instructions": calibration_instructions, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_calibration_instructions_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_calibration_instructions_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drugs_by_carcinogenic_mutagenic_fertility(
    carcinogenic_info: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drugs_by_carcinogenic_mutagenic_fertility",
            "arguments": {
                "carcinogenic_info": carcinogenic_info,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_carcinogenic_mutagenic_fertility_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_carcinogenic_mutagenic_fertility_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_SPL_ID(field_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_drug_name_by_SPL_ID", "arguments": {"field_info": field_info, "limit": limit, "skip": skip}}
    )


@mcp.tool()
def FDA_get_drug_names_by_clinical_pharmacology(
    clinical_pharmacology: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_clinical_pharmacology",
            "arguments": {
                "clinical_pharmacology": clinical_pharmacology,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_clinical_pharmacology_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_clinical_pharmacology_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_clinical_studies(clinical_studies: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_clinical_studies",
            "arguments": {"clinical_studies": clinical_studies, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_clinical_studies_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_clinical_studies_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_contraindications(contraindication_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_contraindications",
            "arguments": {
                "contraindication_info": contraindication_info,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_contraindications_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_contraindications_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_controlled_substance_DEA_schedule(
    controlled_substance_schedule: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_controlled_substance_DEA_schedule",
            "arguments": {
                "controlled_substance_schedule": controlled_substance_schedule,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_controlled_substance_DEA_schedule_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_controlled_substance_DEA_schedule_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_dependence_info(dependence_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_dependence_info",
            "arguments": {"dependence_info": dependence_info, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_dependence_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_dependence_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_disposal_info(disposal_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_disposal_info",
            "arguments": {"disposal_info": disposal_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_disposal_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_disposal_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_dosage_info(dosage_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_dosage_info",
            "arguments": {"dosage_info": dosage_info, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_dosage_forms_and_strengths_info(
    dosage_forms_and_strengths: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_dosage_forms_and_strengths_info",
            "arguments": {
                "dosage_forms_and_strengths": dosage_forms_and_strengths,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_dosage_forms_and_strengths_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_dosage_forms_and_strengths_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_abuse_dependence_info(abuse_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_abuse_dependence_info",
            "arguments": {"abuse_info": abuse_info, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_abuse_dependence_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_abuse_dependence_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_lab_test_interference(
    lab_test_interference: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_lab_test_interference",
            "arguments": {
                "lab_test_interference": lab_test_interference,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_lab_test_interference_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_lab_test_interference_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_drug_interactions(interaction_term: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_drug_interactions",
            "arguments": {"interaction_term": interaction_term, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_interactions_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_interactions_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_effective_time(effective_time: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_effective_time",
            "arguments": {"effective_time": effective_time, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_effective_time_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_effective_time_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_environmental_warning(environmental_warning: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_environmental_warning",
            "arguments": {"environmental_warning": environmental_warning, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_environmental_warning_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_environmental_warning_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_food_safety_warnings(field_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_food_safety_warnings",
            "arguments": {"field_info": field_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_general_precautions(precaution_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_general_precautions",
            "arguments": {"precaution_info": precaution_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_general_precautions_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_general_precautions_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_geriatric_use(geriatric_use: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_geriatric_use",
            "arguments": {"geriatric_use": geriatric_use, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_geriatric_use_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_geriatric_use_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_dear_health_care_provider_letter_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_dear_health_care_provider_letter_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_dear_health_care_provider_letter_info(
    letter_info: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_dear_health_care_provider_letter_info",
            "arguments": {"letter_info": letter_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_health_claim(health_claim: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_health_claim",
            "arguments": {"health_claim": health_claim, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_health_claims_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_health_claims_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_document_id(document_id: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_document_id",
            "arguments": {"document_id": document_id, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_document_id_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_document_id_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_inactive_ingredient(inactive_ingredient: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_inactive_ingredient",
            "arguments": {"inactive_ingredient": inactive_ingredient, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_inactive_ingredient_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_inactive_ingredient_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_indication(indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_indication",
            "arguments": {"indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_indications_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_indications_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_information_for_owners_or_caregivers(
    field_info: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_information_for_owners_or_caregivers",
            "arguments": {"field_info": field_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_information_for_owners_or_caregivers_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_information_for_owners_or_caregivers_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_info_for_patients_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_info_for_patients_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_instructions_for_use(
    instructions_for_use: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_instructions_for_use",
            "arguments": {
                "instructions_for_use": instructions_for_use,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_instructions_for_use_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_instructions_for_use_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_retrieve_drug_name_by_device_use(intended_use_of_the_device: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_retrieve_drug_name_by_device_use",
            "arguments": {"intended_use_of_the_device": intended_use_of_the_device, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_retrieve_device_use_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_retrieve_device_use_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_child_safety_info(child_safety_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_child_safety_info",
            "arguments": {
                "child_safety_info": child_safety_info,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_child_safety_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_child_safety_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_labor_and_delivery_info(labor_and_delivery_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_labor_and_delivery_info",
            "arguments": {"labor_and_delivery_info": labor_and_delivery_info, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_labor_and_delivery_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_labor_and_delivery_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_lab_tests(lab_test_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_lab_tests",
            "arguments": {"lab_test_info": lab_test_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_lab_tests_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_lab_tests_by_drug_name", "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip}}
    )


@mcp.tool()
def FDA_get_mechanism_of_action_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_mechanism_of_action_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_mechanism_of_action(mechanism_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_mechanism_of_action",
            "arguments": {"mechanism_info": mechanism_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_microbiology(microbiology_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_microbiology",
            "arguments": {"microbiology_info": microbiology_info, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_microbiology_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_microbiology_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_nonclinical_toxicology_info(
    toxicology_info: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_nonclinical_toxicology_info",
            "arguments": {"toxicology_info": toxicology_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_nonclinical_toxicology_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_nonclinical_toxicology_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_nonteratogenic_effects(
    nonteratogenic_effects: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_nonteratogenic_effects",
            "arguments": {
                "nonteratogenic_effects": nonteratogenic_effects,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_nonteratogenic_effects_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_nonteratogenic_effects_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_info_for_nursing_mothers(
    nursing_mothers_info: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_info_for_nursing_mothers",
            "arguments": {
                "nursing_mothers_info": nursing_mothers_info,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_info_for_nursing_mothers_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_info_for_nursing_mothers_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_other_safety_info(safety_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_other_safety_info",
            "arguments": {"safety_info": safety_info, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_other_safety_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_other_safety_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_overdosage_info(overdosage_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_overdosage_info",
            "arguments": {"overdosage_info": overdosage_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_overdosage_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_overdosage_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_principal_display_panel(display_panel_content: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_principal_display_panel",
            "arguments": {"display_panel_content": display_panel_content, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_principal_display_panel_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_principal_display_panel_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_retrieve_drug_names_by_patient_medication_info(patient_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_retrieve_drug_names_by_patient_medication_info",
            "arguments": {"patient_info": patient_info, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_retrieve_patient_medication_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_retrieve_patient_medication_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_pediatric_use(pediatric_use_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_pediatric_use",
            "arguments": {
                "pediatric_use_info": pediatric_use_info,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_pediatric_use_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_pediatric_use_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_pharmacodynamics(pharmacodynamics: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_pharmacodynamics",
            "arguments": {"pharmacodynamics": pharmacodynamics, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_pharmacodynamics_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_pharmacodynamics_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_pharmacogenomics(pharmacogenomics: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_pharmacogenomics",
            "arguments": {"pharmacogenomics": pharmacogenomics, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_pharmacogenomics_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_pharmacogenomics_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_pharmacokinetics(pharmacokinetics_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_pharmacokinetics",
            "arguments": {
                "pharmacokinetics_info": pharmacokinetics_info,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_pharmacokinetics_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_pharmacokinetics_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_precautions(precautions: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_precautions",
            "arguments": {"precautions": precautions, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_precautions_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_precautions_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_pregnancy_effects_info(pregnancy_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_pregnancy_effects_info",
            "arguments": {"pregnancy_info": pregnancy_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_pregnancy_effects_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_pregnancy_effects_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_pregnancy_or_breastfeeding_info(pregnancy_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_pregnancy_or_breastfeeding_info",
            "arguments": {"pregnancy_info": pregnancy_info, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_pregnancy_or_breastfeeding_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_pregnancy_or_breastfeeding_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_contact_for_questions_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_contact_for_questions_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_recent_changes_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_recent_changes_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_reference(reference: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_drug_name_by_reference", "arguments": {"reference": reference, "limit": limit, "skip": skip}}
    )


@mcp.tool()
def FDA_get_reference_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_reference_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_residue_warning(residue_warning: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_residue_warning",
            "arguments": {"residue_warning": residue_warning, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_residue_warning_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_residue_warning_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_risk(risk_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_risk",
            "arguments": {"risk_info": risk_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_risk_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_risk_info_by_drug_name", "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip}}
    )


@mcp.tool()
def FDA_get_drug_names_by_route(route: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_route",
            "arguments": {"route": route, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_route_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_route_info_by_drug_name", "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip}}
    )


@mcp.tool()
def FDA_get_drug_names_by_safe_handling_warning(
    safe_handling_warning: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_safe_handling_warning",
            "arguments": {
                "safe_handling_warning": safe_handling_warning,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_safe_handling_warnings_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_safe_handling_warnings_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_set_id(set_id: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_drug_name_by_set_id", "arguments": {"set_id": set_id, "limit": limit, "skip": skip}}
    )


@mcp.tool()
def FDA_get_drug_names_by_spl_indexing_data_elements(
    spl_indexing_data_elements: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_spl_indexing_data_elements",
            "arguments": {
                "spl_indexing_data_elements": spl_indexing_data_elements,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_spl_indexing_data_elements_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_spl_indexing_data_elements_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_medication_guide(medguide_info: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_medication_guide",
            "arguments": {"medguide_info": medguide_info, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_medication_guide_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_medication_guide_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_from_patient_package_insert(patient_package_insert: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_from_patient_package_insert",
            "arguments": {"patient_package_insert": patient_package_insert, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_patient_package_insert_from_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_patient_package_insert_from_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_ingredient(ingredient_name: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_ingredient",
            "arguments": {"ingredient_name": ingredient_name, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_ingredients_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_ingredients_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_spl_unclassified_section_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_spl_unclassified_section_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_stop_use_info(stop_use_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_stop_use_info",
            "arguments": {"stop_use_info": stop_use_info, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_stop_use_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_stop_use_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_storage_and_handling_info(storage_info: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_storage_and_handling_info",
            "arguments": {"storage_info": storage_info, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_storage_and_handling_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_storage_and_handling_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_safety_summary(summary_text: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_safety_summary",
            "arguments": {"summary_text": summary_text, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_safety_summary_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_safety_summary_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_teratogenic_effects(teratogenic_effects: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_teratogenic_effects",
            "arguments": {
                "teratogenic_effects": teratogenic_effects,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_teratogenic_effects_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_teratogenic_effects_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_population_use(population_use: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_population_use",
            "arguments": {"population_use": population_use, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_population_use_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_population_use_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_user_safety_warning_by_drug_names(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_user_safety_warning_by_drug_names",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_user_safety_warning(safety_warning: str, indication: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_user_safety_warning",
            "arguments": {"safety_warning": safety_warning, "indication": indication, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_name_by_warnings(warning_text: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_name_by_warnings",
            "arguments": {"warning_text": warning_text, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_warnings_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_warnings_by_drug_name", "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip}}
    )


@mcp.tool()
def FDA_get_warnings_and_cautions_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_warnings_and_cautions_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_names_by_warnings_and_cautions(
    warnings_and_cautions_info: str, indication: str, limit: int, skip: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_drug_names_by_warnings_and_cautions",
            "arguments": {
                "warnings_and_cautions_info": warnings_and_cautions_info,
                "indication": indication,
                "limit": limit,
                "skip": skip,
            },
        }
    )


@mcp.tool()
def FDA_get_when_using_info(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_when_using_info", "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip}}
    )


@mcp.tool()
def FDA_get_brand_name_generic_name(drug_name: str, limit: int) -> dict:
    return engine.run_one_function(
        {"name": "FDA_get_brand_name_generic_name", "arguments": {"drug_name": drug_name, "limit": limit}}
    )


@mcp.tool()
def FDA_get_do_not_use_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_do_not_use_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_purpose_info_by_drug_name(drug_name: str, limit: int, skip: int) -> dict:
    return engine.run_one_function(
        {
            "name": "FDA_get_purpose_info_by_drug_name",
            "arguments": {"drug_name": drug_name, "limit": limit, "skip": skip},
        }
    )


@mcp.tool()
def FDA_get_drug_generic_name(drug_name: str) -> dict:
    return engine.run_one_function({"name": "FDA_get_drug_generic_name", "arguments": {"drug_name": drug_name}})


@mcp.tool()
def get_joint_associated_diseases_by_HPO_ID_list(HPO_ID_list: list[str], limit: int, offset: int) -> dict:
    return engine.run_one_function(
        {
            "name": "get_joint_associated_diseases_by_HPO_ID_list",
            "arguments": {"HPO_ID_list": HPO_ID_list, "limit": limit, "offset": offset},
        }
    )


@mcp.tool()
def get_phenotype_by_HPO_ID(id: str) -> dict:
    return engine.run_one_function({"name": "get_phenotype_by_HPO_ID", "arguments": {"id": id}})


@mcp.tool()
def get_HPO_ID_by_phenotype(query: str, limit: int, offset: int) -> dict:
    return engine.run_one_function(
        {"name": "get_HPO_ID_by_phenotype", "arguments": {"query": query, "limit": limit, "offset": offset}}
    )


@mcp.tool()
def OpenTargets_get_associated_targets_by_disease_efoId(efoId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_associated_targets_by_disease_efoId", "arguments": {"efoId": efoId}}
    )


@mcp.tool()
def OpenTargets_get_diseases_phenotypes_by_target_ensembl(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_diseases_phenotypes_by_target_ensembl", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_target_disease_evidence(efoId: str, ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_target_disease_evidence", "arguments": {"efoId": efoId, "ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_drug_warnings_by_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_warnings_by_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_drug_mechanisms_of_action_by_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_mechanisms_of_action_by_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_associated_drugs_by_disease_efoId(efoId: str, size: int) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_associated_drugs_by_disease_efoId", "arguments": {"efoId": efoId, "size": size}}
    )


@mcp.tool()
def OpenTargets_get_similar_entities_by_disease_efoId(efoId: str, threshold: float, size: int) -> dict:
    return engine.run_one_function(
        {
            "name": "OpenTargets_get_similar_entities_by_disease_efoId",
            "arguments": {"efoId": efoId, "threshold": threshold, "size": size},
        }
    )


@mcp.tool()
def OpenTargets_get_similar_entities_by_drug_chemblId(chemblId: str, threshold: float, size: int) -> dict:
    return engine.run_one_function(
        {
            "name": "OpenTargets_get_similar_entities_by_drug_chemblId",
            "arguments": {"chemblId": chemblId, "threshold": threshold, "size": size},
        }
    )


@mcp.tool()
def OpenTargets_get_similar_entities_by_target_ensemblID(ensemblId: str, threshold: float, size: int) -> dict:
    return engine.run_one_function(
        {
            "name": "OpenTargets_get_similar_entities_by_target_ensemblID",
            "arguments": {"ensemblId": ensemblId, "threshold": threshold, "size": size},
        }
    )


@mcp.tool()
def OpenTargets_get_associated_phenotypes_by_disease_efoId(efoId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_associated_phenotypes_by_disease_efoId", "arguments": {"efoId": efoId}}
    )


@mcp.tool()
def OpenTargets_get_drug_withdrawn_blackbox_status_by_chemblId(chemblId: list[str]) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_withdrawn_blackbox_status_by_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_search_category_counts_by_query_string(queryString: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_search_category_counts_by_query_string", "arguments": {"queryString": queryString}}
    )


@mcp.tool()
def OpenTargets_get_disease_id_description_by_name(diseaseName: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_disease_id_description_by_name", "arguments": {"diseaseName": diseaseName}}
    )


@mcp.tool()
def OpenTargets_get_drug_id_description_by_name(drugName: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_id_description_by_name", "arguments": {"drugName": drugName}}
    )


@mcp.tool()
def OpenTargets_get_drug_chembId_by_generic_name(drugName: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_chembId_by_generic_name", "arguments": {"drugName": drugName}}
    )


@mcp.tool()
def OpenTargets_get_drug_indications_by_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_indications_by_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_target_gene_ontology_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_gene_ontology_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_target_homologues_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_homologues_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_target_safety_profile_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_safety_profile_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_biological_mouse_models_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_biological_mouse_models_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_target_genomic_location_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_genomic_location_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_target_subcellular_locations_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_subcellular_locations_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_target_synonyms_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_synonyms_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_target_tractability_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_tractability_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_target_classes_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_classes_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_target_enabling_packages_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_enabling_packages_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_target_interactions_by_ensemblID(ensemblId: str, page: dict) -> dict:
    return engine.run_one_function(
        {
            "name": "OpenTargets_get_target_interactions_by_ensemblID",
            "arguments": {"ensemblId": ensemblId, "page": page},
        }
    )


@mcp.tool()
def OpenTargets_get_disease_ancestors_parents_by_efoId(efoId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_disease_ancestors_parents_by_efoId", "arguments": {"efoId": efoId}}
    )


@mcp.tool()
def OpenTargets_get_disease_descendants_children_by_efoId(efoId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_disease_descendants_children_by_efoId", "arguments": {"efoId": efoId}}
    )


@mcp.tool()
def OpenTargets_get_disease_locations_by_efoId(efoId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_disease_locations_by_efoId", "arguments": {"efoId": efoId}}
    )


@mcp.tool()
def OpenTargets_get_disease_synonyms_by_efoId(efoId: str) -> dict:
    return engine.run_one_function({"name": "OpenTargets_get_disease_synonyms_by_efoId", "arguments": {"efoId": efoId}})


@mcp.tool()
def OpenTargets_get_disease_description_by_efoId(efoId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_disease_description_by_efoId", "arguments": {"efoId": efoId}}
    )


@mcp.tool()
def OpenTargets_get_disease_therapeutic_areas_by_efoId(efoId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_disease_therapeutic_areas_by_efoId", "arguments": {"efoId": efoId}}
    )


@mcp.tool()
def OpenTargets_get_drug_adverse_events_by_chemblId(chemblId: str, page: dict) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_adverse_events_by_chemblId", "arguments": {"chemblId": chemblId, "page": page}}
    )


@mcp.tool()
def OpenTargets_get_known_drugs_by_drug_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_known_drugs_by_drug_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_parent_child_molecules_by_drug_chembl_ID(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_parent_child_molecules_by_drug_chembl_ID", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_approved_indications_by_drug_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_approved_indications_by_drug_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_drug_description_by_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_description_by_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_drug_synonyms_by_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_synonyms_by_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_drug_trade_names_by_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_trade_names_by_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_drug_approval_status_by_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_drug_approval_status_by_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_chemical_probes_by_target_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_chemical_probes_by_target_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_drug_pharmacogenomics_data(chemblId: str, page: dict) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_drug_pharmacogenomics_data", "arguments": {"chemblId": chemblId, "page": page}}
    )


@mcp.tool()
def OpenTargets_get_associated_drugs_by_target_ensemblID(ensemblId: str, size: int, cursor: str) -> dict:
    return engine.run_one_function(
        {
            "name": "OpenTargets_get_associated_drugs_by_target_ensemblID",
            "arguments": {"ensemblId": ensemblId, "size": size, "cursor": cursor},
        }
    )


@mcp.tool()
def OpenTargets_get_associated_diseases_by_drug_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_associated_diseases_by_drug_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_get_associated_targets_by_drug_chemblId(chemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_associated_targets_by_drug_chemblId", "arguments": {"chemblId": chemblId}}
    )


@mcp.tool()
def OpenTargets_multi_entity_search_by_query_string(queryString: str, entityNames: list[str], page: dict) -> dict:
    return engine.run_one_function(
        {
            "name": "OpenTargets_multi_entity_search_by_query_string",
            "arguments": {"queryString": queryString, "entityNames": entityNames, "page": page},
        }
    )


@mcp.tool()
def OpenTargets_get_gene_ontology_terms_by_goID(goIds: list[str]) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_gene_ontology_terms_by_goID", "arguments": {"goIds": goIds}}
    )


@mcp.tool()
def OpenTargets_get_target_constraint_info_by_ensemblID(ensemblId: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_constraint_info_by_ensemblID", "arguments": {"ensemblId": ensemblId}}
    )


@mcp.tool()
def OpenTargets_get_publications_by_disease_efoId(
    entityId: str, additionalIds: list[str], startYear: int, startMonth: int, endYear: int, endMonth: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "OpenTargets_get_publications_by_disease_efoId",
            "arguments": {
                "entityId": entityId,
                "additionalIds": additionalIds,
                "startYear": startYear,
                "startMonth": startMonth,
                "endYear": endYear,
                "endMonth": endMonth,
            },
        }
    )


@mcp.tool()
def OpenTargets_get_publications_by_target_ensemblID(
    entityId: str, additionalIds: list[str], startYear: int, startMonth: int, endYear: int, endMonth: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "OpenTargets_get_publications_by_target_ensemblID",
            "arguments": {
                "entityId": entityId,
                "additionalIds": additionalIds,
                "startYear": startYear,
                "startMonth": startMonth,
                "endYear": endYear,
                "endMonth": endMonth,
            },
        }
    )


@mcp.tool()
def OpenTargets_get_publications_by_drug_chemblId(
    entityId: str, additionalIds: list[str], startYear: int, startMonth: int, endYear: int, endMonth: int
) -> dict:
    return engine.run_one_function(
        {
            "name": "OpenTargets_get_publications_by_drug_chemblId",
            "arguments": {
                "entityId": entityId,
                "additionalIds": additionalIds,
                "startYear": startYear,
                "startMonth": startMonth,
                "endYear": endYear,
                "endMonth": endMonth,
            },
        }
    )


@mcp.tool()
def OpenTargets_get_target_id_description_by_name(targetName: str) -> dict:
    return engine.run_one_function(
        {"name": "OpenTargets_get_target_id_description_by_name", "arguments": {"targetName": targetName}}
    )


if __name__ == "__main__":
    mcp.run()
