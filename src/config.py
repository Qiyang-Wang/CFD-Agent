import os
import sys
import yaml
import json
import subprocess

# of_path = "~/OpenFOAM/OpenFOAM-v2406"

def ensure_directory_exists(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # If it doesn't exist, create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' has been created.")

SRC_DIR = "src"
Src_PATH = os.path.dirname(os.path.abspath(__file__))
Base_PATH = os.path.dirname(Src_PATH)

R1_temperature = 0.9
V3_temperature = 0.7

run_time = 3

OpenFOAM_path = ""

OpenFOAM_tutorial_path = ""

all_case_requirement_json = None

all_case_dict = None

target_case_requirement_json = None

def convert_boundary_names_to_lowercase(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if key == "boundaries":
                new_dict[key] = {k.lower(): v for k, v in value.items()}
            else:
                new_dict[key] = convert_boundary_names_to_lowercase(value)
        return new_dict
    elif isinstance(data, list):
        return [convert_boundary_names_to_lowercase(item) for item in data]
    else:
        return data

global_target_case_dict = None

# case_boundaries = []
case_solver = None
case_turbulence_type = "laminar"
case_turbulece_model = "invalid"
case_boundary_names = None

reference_case_searching_round = 10

case_name = None

Database_OFv24_PATH = f'{Base_PATH}/database_OFv24'
TEMP_PATH = f'{Base_PATH}/temp'
OUTPUT_CHATCFD_PATH = f'{Base_PATH}/run_chatcfd'
OUTPUT_PATH = None

case_grid = None
pdf_path = None

ensure_directory_exists(Database_OFv24_PATH)
ensure_directory_exists(OUTPUT_CHATCFD_PATH)
ensure_directory_exists(TEMP_PATH)

paper_case_number = None

paper_content = " "
paper_table = " "

boundary_type_match = None

global_OF_keywords = None

best_reference_cases = []

of_tutorial_dir = '/home/fane/OpenFOAM/fane-v2406/tutorials'

global_file_requirement = {}

global_files = None

pdf_short_case_description = None

OF_data_path = f"{Database_OFv24_PATH}/processed_merged_OF_cases.json"

OF_case_data_dict = {}

max_running_test_round = 30

general_prompts = '''Respond to the following user query in a comprehensive and detailed way. You can write down your thought process before responding. Write your thoughts after “Here is my thought process:” and write your response after “Here is my response:”. \n'''

steady_solvers = ["laplacianFoam", "overLaplacianDyMFoam","potentialFoam","overPotentialFoam","scalarTransportFoam","adjointShapeOptimizationFoam","boundaryFoam","simpleFoam","overSimpleFoam","porousSimpleFoam","SRFSimpleFoam","rhoSimpleFoam","overRhoSimpleFoam","rhoPorousSimpleFoam","interFoam","interMixingFoam","interIsoFoam","interPhaseChangeFoam","MPPICInterFoam","multiphaseInterFoam","potentialFreeSurfaceFoam","potentialFreeSurfaceDyMFoam","buoyantBoussinesqSimpleFoam","buoyantSimpleFoam","chtMultiRegionSimpleFoam","thermoFoam","icoUncoupledKinematicParcelFoam","simpleReactingParcelFoam","simpleCoalParcelFoam","simpleSprayFoam","uncoupledKinematicParcelFoam", "solidEquilibriumDisplacementFoam","financialFoam"]

solver_keywords = ['buoyantBoussinesqSimpleFoam', 'overInterDyMFoam', 'kinematicParcelFoam', 
                   'buoyantSimpleFoam', 'reactingFoam', 'rhoReactingFoam', 
                   'compressibleInterDyMFoam', 'XiEngineFoam', 'pimpleFoam', 
                   'cavitatingFoam', 'adjointOptimisationFoam', 'overPimpleDyMFoam', 
                   'twoPhaseEulerFoam', 'interMixingFoam', 'compressibleInterFoam', 
                   'multiphaseInterFoam', 'porousSimpleFoam', 'overRhoSimpleFoam', 
                   'overLaplacianDyMFoam', 'interFoam', 'MPPICInterFoam', 'icoFoam', 
                   'overRhoPimpleDyMFoam', 'interPhaseChangeDyMFoam', 'sonicDyMFoam', 
                   'chtMultiRegionTwoPhaseEulerFoam', 'adjointShapeOptimizationFoam', 
                   'laplacianFoam', 'fireFoam', 'rhoSimpleFoam', 'overCompressibleInterDyMFoam', 
                   'shallowWaterFoam', 'simpleFoam', 'snappyHexMesh', 'sonicLiquidFoam', 
                   'sonicFoam', 'icoUncoupledKinematicParcelFoam', 'overSimpleFoam', 
                   'driftFluxFoam', 'interIsoFoam', 'uncoupledKinematicParcelDyMFoam', 
                   'sprayFoam', 'buoyantPimpleFoam', 'reactingHeterogenousParcelFoam', 
                   'chemFoam', 'acousticFoam', 'nonNewtonianIcoFoam', 'simpleReactingParcelFoam', 
                   'overInterPhaseChangeDyMFoam', 'boundaryFoam', 'compressibleMultiphaseInterFoam', 
                   'coalChemistryFoam', 'coldEngineFoam', 'rhoPimpleAdiabaticFoam', 
                   'MPPICDyMFoam', 'icoReactingMultiphaseInterFoam', 'SRFPimpleFoam', 
                   'overBuoyantPimpleDyMFoam', 'solidFoam', 'reactingParcelFoam', 
                   'icoUncoupledKinematicParcelDyMFoam', 'compressibleInterIsoFoam', 
                   'potentialFreeSurfaceFoam', 'chtMultiRegionFoam', 'XiDyMFoam', 
                   'multiphaseEulerFoam', 'overPotentialFoam', 'interCondensatingEvaporatingFoam', 
                   'potentialFreeSurfaceDyMFoam', 'subsetMesh', 'twoLiquidMixingFoam', 
                   'rhoPimpleFoam', 'MPPICFoam', 'pisoFoam', 'potentialFoam', 
                   'reactingTwoPhaseEulerFoam', 'reactingMultiphaseEulerFoam', 
                   'rhoPorousSimpleFoam', 'rhoCentralFoam', 'SRFSimpleFoam', 
                   'PDRFoam', 'interPhaseChangeFoam', 'buoyantBoussinesqPimpleFoam', 
                   'XiFoam', 'dnsFoam', 'chtMultiRegionSimpleFoam']

turbulence_type_keywords = ['laminar', 'RAS', 'LES', 'twoPhaseTransport']

turbulence_model_keywords = ['SpalartAllmarasDDES', 'Smagorinsky', 'SpalartAllmaras', 
                                        'SpalartAllmarasIDDES', 'kOmegaSST', 'buoyantKEpsilon', 
                                        'kkLOmega', 'RNGkEpsilon', 'WALE', 'LaunderSharmaKE', 
                                        'realizableKE', 'PDRkEpsilon', 'dynamicKEqn', 
                                        'kOmegaSSTLM', 'kEqn', 'kEpsilon']

boundary_type_keywords = ['overset', 'zeroGradient', 'fixedValue', 'movingWallVelocity', 'inletOutlet', 'symmetryPlane', 'symmetry', 'empty', 'uniformFixedValue', 'noSlip', 'cyclicAMI', 'mappedField', 'calculated', 'waveTransmissive', 'compressible::alphatWallFunction', 'supersonicFreestream', 'epsilonWallFunction', 'kqRWallFunction', 'nutkWallFunction', 'slip', 'turbulentIntensityKineticEnergyInlet', 'turbulentMixingLengthDissipationRateInlet', 'flowRateInletVelocity', 'freestreamPressure', 'omegaWallFunction', 'freestreamVelocity', 'pressureInletOutletVelocity', 'nutUWallFunction', 'totalPressure', 'wedge', 'totalTemperature', 'turbulentInlet', 'fixedMean', 'plenumPressure', 'pressureInletVelocity', 'fluxCorrectedVelocity', 'mixed', 'uniformTotalPressure', 'outletMappedUniformInletHeatAddition', 'clampedPlate', 'nutUSpaldingWallFunction', 'compressible::turbulentTemperatureTwoPhaseRadCoupledMixed', 'copiedFixedValue', 'prghTotalPressure', 'fixedFluxPressure', 'lumpedMassWallTemperature', 'greyDiffusiveRadiation', 'compressible::turbulentTemperatureRadCoupledMixed', 'externalWallHeatFluxTemperature', 'fixedGradient', 'humidityTemperatureCoupledMixed', 'wideBandDiffusiveRadiation', 'greyDiffusiveRadiationViewFactor', 'alphatJayatillekeWallFunction', 'processor', 'compressible::thermalBaffle', 'compressible::alphatJayatillekeWallFunction', 'prghPressure', 'MarshakRadiation', 'surfaceNormalFixedValue', 'turbulentMixingLengthFrequencyInlet', 'interstitialInletVelocity', 'JohnsonJacksonParticleSlip', 'JohnsonJacksonParticleTheta', 'mapped', 'fixedMultiPhaseHeatFlux', 'alphaContactAngle', 'permeableAlphaPressureInletOutletVelocity', 'prghPermeableAlphaTotalPressure', 'nutkRoughWallFunction', 'constantAlphaContactAngle', 'waveAlpha', 'waveVelocity', 'variableHeightFlowRate', 'outletPhaseMeanVelocity', 'variableHeightFlowRateInletVelocity', 'rotatingWallVelocity', 'cyclic', 'porousBafflePressure', 'translatingWallVelocity', 'multiphaseEuler::alphaContactAngle', 'pressureInletOutletParSlipVelocity', 'waveSurfacePressure', 'flowRateOutletVelocity', 'timeVaryingMassSorption', 'adjointOutletPressure', 'adjointOutletVelocity', 'SRFVelocity', 'adjointFarFieldPressure', 'adjointInletVelocity', 'adjointWallVelocity', 'adjointInletNuaTilda', 'adjointOutletNuaTilda', 'nutLowReWallFunction', 'outletInlet', 'freestream', 'adjointFarFieldVelocity', 'adjointFarFieldNuaTilda', 'waWallFunction', 'adjointZeroInlet', 'adjointOutletWa', 'kaqRWallFunction', 'adjointOutletKa', 'adjointFarFieldTMVar2', 'adjointFarFieldTMVar1', 'adjointOutletVelocityFlux', 'adjointOutletNuaTildaFlux', 'SRFFreestreamVelocity', 'timeVaryingMappedFixedValue', 'atmBoundaryLayerInletVelocity', 'atmBoundaryLayerInletEpsilon', 'atmBoundaryLayerInletK', 'atmNutkWallFunction', 'nutUBlendedWallFunction', 'maxwellSlipU', 'smoluchowskiJumpT', 'freeSurfacePressure', 'freeSurfaceVelocity']

boundary_required_field = [{'type': 'overset', 'require_entry': ['value']}, {'type': 'zeroGradient', 'require_entry': ['value']}, {'type': 'fixedValue', 'require_entry': []}, {'type': 'movingWallVelocity', 'require_entry': []}, {'type': 'inletOutlet', 'require_entry': []}, {'type': 'symmetryPlane', 'require_entry': ['value']}, {'type': 'symmetry', 'require_entry': []}, {'type': 'empty', 'require_entry': []}, {'type': 'uniformFixedValue', 'require_entry': []}, {'type': 'noSlip', 'require_entry': []}, {'type': 'cyclicAMI', 'require_entry': []}, {'type': 'mappedField', 'require_entry': []}, {'type': 'calculated', 'require_entry': []}, {'type': 'waveTransmissive', 'require_entry': []}, {'type': 'compressible::alphatWallFunction', 'require_entry': ['Prt']}, {'type': 'supersonicFreestream', 'require_entry': []}, {'type': 'epsilonWallFunction', 'require_entry': []}, {'type': 'kqRWallFunction', 'require_entry': []}, {'type': 'nutkWallFunction', 'require_entry': []}, {'type': 'slip', 'require_entry': ['value']}, {'type': 'turbulentIntensityKineticEnergyInlet', 'require_entry': []}, {'type': 'turbulentMixingLengthDissipationRateInlet', 'require_entry': []}, {'type': 'flowRateInletVelocity', 'require_entry': ['massFlowRate', 'value']}, {'type': 'freestreamPressure', 'require_entry': []}, {'type': 'omegaWallFunction', 'require_entry': []}, {'type': 'freestreamVelocity', 'require_entry': []}, {'type': 'pressureInletOutletVelocity', 'require_entry': []}, {'type': 'nutUWallFunction', 'require_entry': []}, {'type': 'totalPressure', 'require_entry': []}, {'type': 'wedge', 'require_entry': []}, {'type': 'totalTemperature', 'require_entry': []}, {'type': 'turbulentInlet', 'require_entry': []}, {'type': 'fixedMean', 'require_entry': []}, {'type': 'plenumPressure', 'require_entry': []}, {'type': 'pressureInletVelocity', 'require_entry': []}, {'type': 'fluxCorrectedVelocity', 'require_entry': []}, {'type': 'mixed', 'require_entry': []}, {'type': 'uniformTotalPressure', 'require_entry': []}, {'type': 'outletMappedUniformInletHeatAddition', 'require_entry': []}, {'type': 'clampedPlate', 'require_entry': []}, {'type': 'nutUSpaldingWallFunction', 'require_entry': []}, {'type': 'compressible::turbulentTemperatureTwoPhaseRadCoupledMixed', 'require_entry': []}, {'type': 'copiedFixedValue', 'require_entry': []}, {'type': 'prghTotalPressure', 'require_entry': []}, {'type': 'fixedFluxPressure', 'require_entry': []}, {'type': 'lumpedMassWallTemperature', 'require_entry': []}, {'type': 'greyDiffusiveRadiation', 'require_entry': []}, {'type': 'compressible::turbulentTemperatureRadCoupledMixed', 'require_entry': ['qr', 'qrNbr', 'kappa']}, {'type': 'externalWallHeatFluxTemperature', 'require_entry': ['q', 'kappaName']}, {'type': 'fixedGradient', 'require_entry': []}, {'type': 'humidityTemperatureCoupledMixed', 'require_entry': []}, {'type': 'wideBandDiffusiveRadiation', 'require_entry': []}, {'type': 'greyDiffusiveRadiationViewFactor', 'require_entry': []}, {'type': 'alphatJayatillekeWallFunction', 'require_entry': []}, {'type': 'processor', 'require_entry': ['value']}, {'type': 'compressible::thermalBaffle', 'require_entry': []}, {'type': 'compressible::alphatJayatillekeWallFunction', 'require_entry': []}, {'type': 'prghPressure', 'require_entry': []}, {'type': 'MarshakRadiation', 'require_entry': []}, {'type': 'surfaceNormalFixedValue', 'require_entry': ['value']}, {'type': 'turbulentMixingLengthFrequencyInlet', 'require_entry': ['k']}, {'type': 'interstitialInletVelocity', 'require_entry': []}, {'type': 'JohnsonJacksonParticleSlip', 'require_entry': []}, {'type': 'JohnsonJacksonParticleTheta', 'require_entry': []}, {'type': 'mapped', 'require_entry': []}, {'type': 'fixedMultiPhaseHeatFlux', 'require_entry': []}, {'type': 'alphaContactAngle', 'require_entry': []}, {'type': 'permeableAlphaPressureInletOutletVelocity', 'require_entry': []}, {'type': 'prghPermeableAlphaTotalPressure', 'require_entry': []}, {'type': 'nutkRoughWallFunction', 'require_entry': []}, {'type': 'constantAlphaContactAngle', 'require_entry': []}, {'type': 'waveAlpha', 'require_entry': []}, {'type': 'waveVelocity', 'require_entry': []}, {'type': 'variableHeightFlowRate', 'require_entry': []}, {'type': 'outletPhaseMeanVelocity', 'require_entry': []}, {'type': 'variableHeightFlowRateInletVelocity', 'require_entry': []}, {'type': 'rotatingWallVelocity', 'require_entry': []}, {'type': 'cyclic', 'require_entry': ['value']}, {'type': 'porousBafflePressure', 'require_entry': []}, {'type': 'translatingWallVelocity', 'require_entry': []}, {'type': 'multiphaseEuler::alphaContactAngle', 'require_entry': []}, {'type': 'pressureInletOutletParSlipVelocity', 'require_entry': []}, {'type': 'waveSurfacePressure', 'require_entry': []}, {'type': 'flowRateOutletVelocity', 'require_entry': []}, {'type': 'timeVaryingMassSorption', 'require_entry': []}, {'type': 'adjointOutletPressure', 'require_entry': []}, {'type': 'adjointOutletVelocity', 'require_entry': []}, {'type': 'SRFVelocity', 'require_entry': []}, {'type': 'adjointFarFieldPressure', 'require_entry': []}, {'type': 'adjointInletVelocity', 'require_entry': []}, {'type': 'adjointWallVelocity', 'require_entry': []}, {'type': 'adjointInletNuaTilda', 'require_entry': []}, {'type': 'adjointOutletNuaTilda', 'require_entry': []}, {'type': 'nutLowReWallFunction', 'require_entry': []}, {'type': 'outletInlet', 'require_entry': []}, {'type': 'freestream', 'require_entry': []}, {'type': 'adjointFarFieldVelocity', 'require_entry': []}, {'type': 'adjointFarFieldNuaTilda', 'require_entry': []}, {'type': 'waWallFunction', 'require_entry': []}, {'type': 'adjointZeroInlet', 'require_entry': []}, {'type': 'adjointOutletWa', 'require_entry': []}, {'type': 'kaqRWallFunction', 'require_entry': []}, {'type': 'adjointOutletKa', 'require_entry': []}, {'type': 'adjointFarFieldTMVar2', 'require_entry': []}, {'type': 'adjointFarFieldTMVar1', 'require_entry': []}, {'type': 'adjointOutletVelocityFlux', 'require_entry': []}, {'type': 'adjointOutletNuaTildaFlux', 'require_entry': []}, {'type': 'SRFFreestreamVelocity', 'require_entry': []}, {'type': 'timeVaryingMappedFixedValue', 'require_entry': []}, {'type': 'atmBoundaryLayerInletVelocity', 'require_entry': []}, {'type': 'atmBoundaryLayerInletEpsilon', 'require_entry': []}, {'type': 'atmBoundaryLayerInletK', 'require_entry': []}, {'type': 'atmNutkWallFunction', 'require_entry': []}, {'type': 'nutUBlendedWallFunction', 'require_entry': []}, {'type': 'maxwellSlipU', 'require_entry': []}, {'type': 'smoluchowskiJumpT', 'require_entry': []}, {'type': 'freeSurfacePressure', 'require_entry': []}, {'type': 'freeSurfaceVelocity', 'require_entry': []}]

thermodynamic_model_keywords = ['hePsiThermo', 'heRhoThermo', 'heSolidThermo']

string_of_turbulence_type_keywords= ", ".join(turbulence_type_keywords)
string_of_turbulence_model= ", ".join(turbulence_model_keywords)
string_of_solver_keywords = ", ".join(solver_keywords)
string_of_boundary_type_keywords = ", ".join(boundary_type_keywords)
string_of_thermodynamic_model = ", ".join(thermodynamic_model_keywords)

case_log_write = False

flag_OF_tutorial_processed = False

error_history = []

mesh_convert_success = False

set_controlDict_time = False

pdf_chunk_d = 1.5

case_ic_bc_from_paper = ""