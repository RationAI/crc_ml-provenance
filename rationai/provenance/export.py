# Standard Imports
from pathlib import Path
from collections import defaultdict
import argparse
import json

# Third-party Imports
import prov.model as prov

# Local Imports
from rationai.utils.provenance import export_to_file
from rationai.utils.provenance import export_to_datacite
from rationai.provenance import ORGANISATION_DOI
from rationai.provenance import PID_NAMESPACE_URI
from rationai.provenance import NAMESPACE_COMMON_MODEL
from rationai.provenance import NAMESPACE_PROV
from rationai.provenance import OUTPUT_DIR
from rationai.provenance import BUNDLE_META
from rationai.provenance import PROVN_NAMESPACE
from rationai.provenance import DOI_NAMESPACE


def filter_entity(entity: prov.ProvEntity):
    '''Keep only connector type entities.'''
    FILTER_TYPES = [
        f'{NAMESPACE_COMMON_MODEL}:externalInputConnector',
        f'{NAMESPACE_COMMON_MODEL}:forwardConnector',
        f'{NAMESPACE_COMMON_MODEL}:currentConnector'
    ]
    prov_type = list(entity.get_attribute(f'{NAMESPACE_PROV}:type'))
    if prov_type:
        return str(prov_type[0]) in FILTER_TYPES
    else:
        return False
    
    
def deserialize_jsonprov_dir(prov_dir: Path) -> dict:
    '''Read and parse all PROV-JSONs in the directory.
    Group connector entities by name.'''
    connector_entities = defaultdict(list)
    
    json_prov_files = prov_dir.glob('*.json')
    for json_prov_file in json_prov_files:
        with json_prov_file.open('r') as json_f:
            prov_doc = prov.ProvDocument.deserialize(json_f, format='json')
        connector_entities = process_saved_prov(prov_doc, connector_entities)
    return connector_entities


def attribute_values_to_qualnames(doc: prov.ProvDocument, entity: prov.ProvEntity):
    '''Attribute values are normally saved as strings. 
    For a namespace to transfer to new doc it must be
    saved as QualifiedName.'''
    for attr_name, attr_vals in entity._attributes.items():
        new_attr_vals = []
        for attr_val in attr_vals:
            new_attr_val = doc.valid_qualified_name(attr_val)
            new_attr_vals.append(new_attr_val)
        entity._attributes[attr_name] = set(new_attr_vals)
    return entity


def process_saved_prov(doc: prov.ProvDocument, conn_map: dict) -> dict:
    '''Populate mapping structure with filtered elements.'''
    for bundle in doc.bundles:
        for conn_entity in filter(filter_entity, bundle.records):
            if f'{NAMESPACE_COMMON_MODEL}:forwardConnector' in [str(attr) for attr in conn_entity.get_attribute(f'{NAMESPACE_PROV}:type')]:
                backward_entity = create_backward_entity(bundle, conn_entity)
                backward_entity = attribute_values_to_qualnames(doc, backward_entity)
                conn_map[conn_entity.identifier.localpart].append(backward_entity)
                pass
            conn_entity = attribute_values_to_qualnames(doc, conn_entity)
            conn_map[conn_entity.identifier.localpart].append(conn_entity)
    return conn_map


def create_backward_entity(bundle: prov.ProvBundle, entity: prov.ProvEntity):
    '''Creates backward-linking entity for PID'''
    entity_identifier = entity.identifier.localpart
    remote_entity = bundle.entity(f"{DOI_NAMESPACE}:{entity_identifier}", other_attributes={
        f'{NAMESPACE_PROV}:type': f'{NAMESPACE_COMMON_MODEL}:backwardConnector',
        f'{NAMESPACE_COMMON_MODEL}:senderBundleId': entity.bundle.identifier,
        #f'{NAMESPACE_COMMON_MODEL}:senderServiceUri': PROVN_NAMESPACE,
        f'{NAMESPACE_COMMON_MODEL}:metabundle': f'{PROVN_NAMESPACE}:{BUNDLE_META}'
    })
    return remote_entity


def convert_to_docs(conn_map: dict) -> list:
    '''Create new ProvDocument containing only Connector records.'''
    doc_map = {}
    for connector_name, entities in conn_map.items():
        doc = prov.ProvDocument()
        for entity in entities:
            doc.add_record(entity)
        doc_map[connector_name] = doc
    return doc_map
            

def export_docs(doc_map: dict, to_datacite: bool):
    '''Save document to PID directory.'''
    pid_directory = OUTPUT_DIR / 'pid'
    if not pid_directory.exists():
        pid_directory.mkdir(parents=True)
    
    for connector_name, prov_doc in doc_map.items():
        export_to_file(prov_doc, (pid_directory / connector_name).with_suffix('.provn'), format='provn')
        export_to_file(prov_doc, (pid_directory / connector_name).with_suffix('.json'), format='json')
    
        if to_datacite:
            export_to_datacite(
                organisation_doi=ORGANISATION_DOI,
                entity_identifier=connector_name,
                remote_git_repo_path=PID_NAMESPACE_URI + connector_name + '.provn'
            )


def resolve_provenance(prov_dir: Path):
    connector_entities = deserialize_jsonprov_dir(prov_dir)
    mapped_docs = convert_to_docs(connector_entities)
    export_docs(mapped_docs, True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required arguments
    parser.add_argument('--config_fp', type=Path, required=True, help='Configuration file')
    parser.add_argument('--eid', type=str, required=True, help='Experiment UUID')
    args = parser.parse_args()
    
    with args.config_fp.open('r') as json_in:
        cfg = json.load(json_in)
        
    experiment_dir = Path(cfg['output_dir']) / args.eid / 'json'
    experiment_dir = OUTPUT_DIR / 'json'
    assert experiment_dir.exists()
    
    resolve_provenance(experiment_dir)
    