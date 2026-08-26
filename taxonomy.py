from __future__ import annotations

"""
TRACE-CPS taxonomy specification.

The 18 leaf properties and the high-level definitions below are grounded in the
paper's proposed verification-oriented taxonomy.  The extra prototype examples
and boundary rules are classifier aids; they are not additional taxonomy labels.
"""

TAXONOMY = {
    "Functional Correctness": [
        "Safety", "Liveness", "Reachability", "Timing Constraints", "Hybrid Dynamics"
    ],
    "Information Protection": [
        "Confidentiality", "Integrity", "Availability", "Authenticity", "Authorization",
        "Accountability", "Non-repudiation"
    ],
    "Operational Assurance": [
        "Privacy", "Reliability", "Resilience", "Recoverability", "Compliance", "Explainability"
    ],
}

PARENT = {p: parent for parent, props in TAXONOMY.items() for p in props}
PROPERTIES = [p for props in TAXONOMY.values() for p in props]

# -----------------------------------------------------------------------------
# Paper-grounded definitions.
# These are consequence/property definitions, not attack-type definitions.
# -----------------------------------------------------------------------------
PAPER_DEFINITIONS = {
    "Safety": (
        "Safety requires that hazardous or undesirable system states never occur. "
        "A violation is supported when the cyber incident actually causes unsafe physical behavior, "
        "a hazardous operating condition, physical harm, or a dangerous control outcome."
    ),
    "Liveness": (
        "Liveness requires that required desirable behavior eventually occurs. A violation is supported "
        "when a required command, task, process, shutdown, recovery step, or normal system progression "
        "is actually prevented from eventually completing."
    ),
    "Reachability": (
        "Reachability concerns whether a particular system state or configuration can actually be reached. "
        "Security-relevant violations include an attacker obtaining a protected privilege, commands reaching "
        "a protected controller, unauthorized access to a critical component, or malware crossing segmentation."
    ),
    "Timing Constraints": (
        "Timing constraints require control actions, sensor updates, synchronization messages, and emergency "
        "responses to occur within required timing bounds. A violation requires an actually documented missed "
        "deadline, excessive latency, stale timing, delayed time-critical response, or synchronization failure."
    ),
    "Hybrid Dynamics": (
        "Hybrid dynamics concerns the interaction of discrete digital control with continuous physical evolution. "
        "A violation requires evidence that the cyber incident actually altered or destabilized continuous physical "
        "behavior such as motion, pressure, flow, frequency, trajectory, process dynamics, or a control loop."
    ),
    "Confidentiality": (
        "Confidentiality requires sensitive information to be disclosed only to authorized entities. A violation "
        "requires actual unauthorized disclosure, exposure, leakage, theft, or exfiltration of sensitive information."
    ),
    "Integrity": (
        "Integrity requires data and commands to remain accurate, complete, and free from unauthorized modification. "
        "A violation includes actual falsification or unauthorized modification of sensor measurements, commands, "
        "PLC logic, firmware, software, configuration, or parameters."
    ),
    "Availability": (
        "Availability requires authorized users, controllers, and applications to retain timely access to required "
        "computational and communication resources. A violation requires an actual loss of access or service that "
        "affects the required system or operational behavior."
    ),
    "Authenticity": (
        "Authenticity requires communicating entities to genuinely possess the identities they claim. A violation "
        "requires successful spoofing, impersonation, forged identity or credential acceptance, or a malicious device, "
        "signal, message, or update being accepted as trusted."
    ),
    "Authorization": (
        "Authorization specifies which authenticated entities may perform particular actions. A violation requires an "
        "entity to actually perform an operation outside its permitted privileges, such as unauthorized control action, "
        "privilege escalation, or unauthorized configuration change."
    ),
    "Accountability": (
        "Accountability requires actions to remain attributable to identifiable entities using trustworthy logs and "
        "audit trails. A violation requires actual loss, destruction, corruption, or untrustworthiness of the records "
        "needed to reconstruct actions or assign responsibility."
    ),
    "Non-repudiation": (
        "Non-repudiation requires evidence that prevents participants from later denying actions they performed. "
        "A violation requires actual loss or invalidation of signatures, authenticated records, or cryptographic/audit "
        "evidence needed to prove a specific action."
    ),
    "Privacy": (
        "Privacy governs the collection, storage, processing, use, and dissemination of personal or sensitive personal "
        "information. A violation requires actual unauthorized exposure, use, processing, or dissemination of PII, "
        "medical, location, behavioral, or similar personal data."
    ),
    "Reliability": (
        "Reliability is the ability of a system to continuously and predictably perform its intended functions under "
        "expected operating conditions. A violation requires actual inconsistent, unstable, repeated, or otherwise "
        "undependable performance of the intended function."
    ),
    "Resilience": (
        "Resilience is the ability to withstand disruption while maintaining essential functionality in an adversarial "
        "environment. A violation requires evidence that essential functionality could not be maintained during the "
        "incident, rather than merely that an attack occurred."
    ),
    "Recoverability": (
        "Recoverability is the ability to restore normal trusted operation after a fault or cyber attack. A violation "
        "requires actual impaired, failed, or materially delayed restoration, reconstruction, validation, or safe return "
        "to normal operation."
    ),
    "Compliance": (
        "Compliance requires system behavior to remain consistent with applicable standards, legislation, certification, "
        "and mandatory regulatory requirements. A violation requires a documented actual non-compliance finding or "
        "breach of an applicable mandatory requirement, not merely the existence of regulation."
    ),
    "Explainability": (
        "Explainability concerns whether important autonomous or system decisions can be justified, reconstructed, and "
        "understood by engineers, operators, regulators, or certification authorities. A violation requires an actually "
        "documented inability to explain or reconstruct a relevant decision or reasoning process."
    ),
}

# -----------------------------------------------------------------------------
# Contrastive prototypes.
# Positive examples express actual documented violations.
# Hard negatives capture the most common confusions observed in testing.
# These are classifier prototypes, not new labels.
# -----------------------------------------------------------------------------
POSITIVE_PROTOTYPES = {
    "Safety": [
        "The cyber action caused physical damage or an unsafe physical condition.",
        "The controller manipulation caused the physical process to enter a hazardous state.",
        "The attack caused unsafe switching, unsafe dosing, collision risk, or physical harm.",
    ],
    "Liveness": [
        "A required command or task could not eventually complete.",
        "The incident indefinitely prevented required normal system progression.",
        "A critical shutdown, production task, or control action failed to complete.",
    ],
    "Reachability": [
        "The attacker successfully reached a protected controller, host, privilege, or network zone.",
        "Malicious commands successfully reached a protected control component.",
        "Malware propagated into a segmented or protected operational environment.",
    ],
    "Timing Constraints": [
        "The incident caused a missed deadline or delayed time-critical control action.",
        "A synchronization failure or excessive latency violated a required timing bound.",
        "Stale sensor information or delayed emergency response violated temporal requirements.",
    ],
    "Hybrid Dynamics": [
        "The cyber manipulation changed or destabilized continuous physical process dynamics.",
        "The attack altered trajectory, pressure, flow, frequency, speed, or a physical control loop.",
        "Digital manipulation caused unintended continuous physical evolution over time.",
    ],
    "Confidentiality": [
        "Sensitive information was actually stolen, leaked, exposed, disclosed, or exfiltrated.",
        "An unauthorized party obtained protected operational or proprietary information.",
        "Protected credentials, medical records, process data, or confidential files were disclosed.",
    ],
    "Integrity": [
        "Data, commands, software, firmware, configuration, or control logic were modified without authorization.",
        "Sensor measurements or control values were falsified or corrupted.",
        "PLC logic, firmware, software, or parameters were maliciously altered.",
    ],
    "Availability": [
        "A required system or service became unavailable to authorized users or processes.",
        "Operators lost access to a required controller, application, network, or communication resource.",
        "The incident caused an actual outage, lockout, or operational service interruption.",
    ],
    "Authenticity": [
        "A forged or spoofed identity, credential, signal, message, device, or update was accepted as genuine.",
        "The attacker successfully impersonated a trusted operator, device, sensor, or service.",
        "A malicious component masqueraded as a trusted component and was accepted.",
    ],
    "Authorization": [
        "An entity actually performed an operation that it was not permitted to perform.",
        "The attacker successfully escalated privileges or executed an unauthorized control action.",
        "A configuration or maintenance action was performed without the required permission.",
    ],
    "Accountability": [
        "Logs or audit records were destroyed, missing, corrupted, or untrustworthy so actions could not be attributed.",
        "The incident prevented investigators from reconstructing who performed important actions.",
        "Trustworthy event recording failed and responsibility could not be assigned.",
    ],
    "Non-repudiation": [
        "Signatures or authenticated records needed to prove an action were invalid or unavailable.",
        "The incident undermined cryptographic proof of who issued a command or performed an action.",
        "A participant could deny an action because trustworthy proof of that action was lost.",
    ],
    "Privacy": [
        "Personal or sensitive personal information was actually exposed, misused, or disseminated without authorization.",
        "Medical, location, behavioral, identity, or other PII was compromised.",
        "Personal information was processed or shared in a way that violated privacy requirements.",
    ],
    "Reliability": [
        "The system actually became inconsistent, unstable, or unable to perform its intended function dependably.",
        "The incident caused repeated malfunction or unpredictable intended-system behavior.",
        "Dependable sensor, communication, production, or navigation behavior was degraded.",
    ],
    "Resilience": [
        "During the attack, the system failed to maintain essential functionality.",
        "The system could not withstand the disruption with graceful degradation or continuity.",
        "Essential service continuity failed while the adversarial disruption was ongoing.",
    ],
    "Recoverability": [
        "Restoration of normal trusted operation was materially delayed or failed.",
        "Recovery required prolonged rebuild, restoration, or validation before safe operation resumed.",
        "Services or physical processes could not be promptly restored after the incident.",
    ],
    "Compliance": [
        "A regulator or authoritative source documented violation of an applicable mandatory requirement.",
        "The incident produced a documented breach of a legal, regulatory, certification, or mandatory standard obligation.",
        "An applicable compliance requirement was actually not satisfied.",
    ],
    "Explainability": [
        "A relevant automated or system decision could not be explained or reconstructed by responsible humans.",
        "The reasoning behind a security-relevant or operational decision was opaque and could not be justified.",
        "Operators or regulators could not obtain an understandable rationale for an important system decision.",
    ],
}

HARD_NEGATIVE_PROTOTYPES = {
    "Safety": [
        "Malware infected systems but no physical damage or unsafe condition occurred.",
        "The attack had the capability to cause physical harm but did not actually do so.",
        "An IT service was disrupted without a documented unsafe physical consequence.",
    ],
    "Liveness": [
        "A service was temporarily unavailable but required tasks later completed normally.",
        "The incident slowed operation without preventing eventual completion.",
        "A hypothetical attack could block progress but no actual task was prevented.",
    ],
    "Reachability": [
        "A vulnerability could allow access but no protected system was actually reached.",
        "Researchers demonstrated a possible attack path without real exploitation.",
        "Malware existed on an IT host without evidence that a protected controller or privilege was reached.",
    ],
    "Timing Constraints": [
        "The service was unavailable but no timing deadline or synchronization requirement was documented.",
        "Operations were disrupted without evidence of a missed time bound.",
        "A delay was possible but no actual timing requirement was violated.",
    ],
    "Hybrid Dynamics": [
        "Physical damage occurred but the source does not document altered continuous dynamics or a control loop.",
        "A controller was compromised without evidence that pressure, flow, speed, trajectory, frequency, or continuous evolution changed.",
        "The incident was an architectural or access failure without documented continuous physical dynamics.",
    ],
    "Confidentiality": [
        "A system was compromised but no sensitive information was disclosed or stolen.",
        "Data could have been accessed but exposure was not confirmed.",
        "The service was unavailable without any information disclosure.",
    ],
    "Integrity": [
        "Malware infected a system but no data, command, software, firmware, configuration, or logic modification was documented.",
        "An attacker gained access but no unauthorized modification was reported.",
        "A service outage occurred without evidence of falsification or modification.",
    ],
    "Availability": [
        "Systems were infected or accessed but required services remained operational.",
        "The attack could shut down a service but no actual outage occurred.",
        "There was no service disruption, outage, lockout, or loss of authorized access.",
    ],
    "Authenticity": [
        "An attacker used unauthorized access without impersonating or spoofing a trusted identity.",
        "Privileges were misused but no false identity, device, signal, or message was accepted as genuine.",
        "A vulnerability could enable spoofing but no successful spoofing occurred.",
    ],
    "Authorization": [
        "A protected host was reached but no out-of-permission action was documented.",
        "Authentication failed or credentials were probed without successful unauthorized action.",
        "A vulnerability could permit excessive privileges but they were not actually exercised.",
    ],
    "Accountability": [
        "The attacker was unknown, but trustworthy logs and audit evidence were not shown to be missing or corrupted.",
        "Attribution to a nation or group was uncertain, without evidence of an audit-trail failure.",
        "An intrusion occurred but logging remained available.",
    ],
    "Non-repudiation": [
        "The attacker was not identified, but there was no failure of signatures or proof of performed actions.",
        "Logs were incomplete without evidence that a participant could deny a specific action.",
        "An authenticity or accountability issue occurred without loss of non-repudiation evidence.",
    ],
    "Privacy": [
        "Confidential operational or proprietary information was exposed, but no personal information was involved.",
        "A system was compromised without exposure or misuse of PII, medical, location, or behavioral data.",
        "Personal data could have been exposed but no actual privacy violation was confirmed.",
    ],
    "Reliability": [
        "A one-time service outage occurred without evidence of inconsistent or unpredictable intended-function performance.",
        "Essential functions were temporarily unavailable but no repeated or unstable behavior was documented.",
        "The attack was resisted successfully and intended functions remained dependable.",
    ],
    "Resilience": [
        "An attack occurred but essential functionality was maintained throughout the disruption.",
        "A non-essential IT service was affected while core operational functions continued.",
        "Recovery took time after the incident, but essential functionality was maintained during the attack.",
    ],
    "Recoverability": [
        "A service was unavailable during the attack but normal operation was promptly restored.",
        "The incident disrupted operation without documented difficulty restoring trusted operation.",
        "Recovery could be difficult, but no actual delayed or failed restoration was reported.",
    ],
    "Compliance": [
        "The organization operates under regulation, but no actual compliance violation was documented.",
        "Personal data was exposed without an explicit regulatory or mandatory non-compliance finding in the evidence.",
        "The incident could trigger regulatory obligations but no violation was established.",
    ],
    "Explainability": [
        "The attacker was unknown or the incident cause was unclear, but no opaque automated decision was involved.",
        "Investigators lacked details about the attack without evidence that a system decision itself was unexplainable.",
        "An AI-enabled system existed but no actual explanation failure was documented.",
    ],
}

# Four evidence-status prototypes.  CONFIRMED must win after the hard-negative
# contrast and actuality checks before a property enters scientific statistics.
STATUS_PROTOTYPES = {
    prop: {
        "CONFIRMED": f"The documented incident actually violated {prop}: {PAPER_DEFINITIONS[prop]}",
        "POTENTIAL": f"The evidence only describes a capability, possibility, demonstration, or hypothetical risk related to {prop}; no actual violation is documented.",
        "UNAFFECTED": f"The evidence explicitly says that the relevant {prop} consequence did not occur or remained unaffected.",
        "CLAIMED": f"A party alleges or claims a {prop} violation, but the evidence does not independently establish it as fact.",
    }
    for prop in PROPERTIES
}

# Retrieval text is built from the paper definition + positive prototypes.  It is
# intentionally semantic (Transformer embeddings), not TF-IDF.
PROPERTY_SPECS = {
    prop: {
        "definition": PAPER_DEFINITIONS[prop],
        "retrieval": " ".join([PAPER_DEFINITIONS[prop], *POSITIVE_PROTOTYPES[prop]]),
        "confirmed": STATUS_PROTOTYPES[prop]["CONFIRMED"],
        "potential": STATUS_PROTOTYPES[prop]["POTENTIAL"],
        "unaffected": STATUS_PROTOTYPES[prop]["UNAFFECTED"],
        "claimed": STATUS_PROTOTYPES[prop]["CLAIMED"],
    }
    for prop in PROPERTIES
}
