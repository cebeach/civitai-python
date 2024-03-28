# coding: utf-8

"""
    Civitai Orchestration Consumer API

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)

    The version of the OpenAPI document: v1
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


from __future__ import annotations
import pprint
import re  # noqa: F401
import json

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from typing import Any, ClassVar, Dict, List, Optional
from civitai.models.module_handle import ModuleHandle
from typing import Optional, Set
from typing_extensions import Self

class Module(BaseModel):
    """
    Module
    """ # noqa: E501
    assembly: Optional[Assembly] = None
    fully_qualified_name: Optional[StrictStr] = Field(default=None, alias="fullyQualifiedName")
    name: Optional[StrictStr] = None
    md_stream_version: Optional[StrictInt] = Field(default=None, alias="mdStreamVersion")
    module_version_id: Optional[StrictStr] = Field(default=None, alias="moduleVersionId")
    scope_name: Optional[StrictStr] = Field(default=None, alias="scopeName")
    module_handle: Optional[ModuleHandle] = Field(default=None, alias="moduleHandle")
    custom_attributes: Optional[List[CustomAttributeData]] = Field(default=None, alias="customAttributes")
    metadata_token: Optional[StrictInt] = Field(default=None, alias="metadataToken")
    __properties: ClassVar[List[str]] = ["assembly", "fullyQualifiedName", "name", "mdStreamVersion", "moduleVersionId", "scopeName", "moduleHandle", "customAttributes", "metadataToken"]

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        protected_namespaces=(),
    )


    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.model_dump(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        # TODO: pydantic v2: use .model_dump_json(by_alias=True, exclude_unset=True) instead
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> Optional[Self]:
        """Create an instance of Module from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        """Return the dictionary representation of the model using alias.

        This has the following differences from calling pydantic's
        `self.model_dump(by_alias=True)`:

        * `None` is only added to the output dict for nullable fields that
          were set at model initialization. Other fields with value `None`
          are ignored.
        * OpenAPI `readOnly` fields are excluded.
        * OpenAPI `readOnly` fields are excluded.
        * OpenAPI `readOnly` fields are excluded.
        * OpenAPI `readOnly` fields are excluded.
        * OpenAPI `readOnly` fields are excluded.
        * OpenAPI `readOnly` fields are excluded.
        * OpenAPI `readOnly` fields are excluded.
        """
        excluded_fields: Set[str] = set([
            "fully_qualified_name",
            "name",
            "md_stream_version",
            "module_version_id",
            "scope_name",
            "custom_attributes",
            "metadata_token",
        ])

        _dict = self.model_dump(
            by_alias=True,
            exclude=excluded_fields,
            exclude_none=True,
        )
        # override the default output from pydantic by calling `to_dict()` of assembly
        if self.assembly:
            _dict['assembly'] = self.assembly.to_dict()
        # override the default output from pydantic by calling `to_dict()` of module_handle
        if self.module_handle:
            _dict['moduleHandle'] = self.module_handle.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each item in custom_attributes (list)
        _items = []
        if self.custom_attributes:
            for _item in self.custom_attributes:
                if _item:
                    _items.append(_item.to_dict())
            _dict['customAttributes'] = _items
        # set to None if fully_qualified_name (nullable) is None
        # and model_fields_set contains the field
        if self.fully_qualified_name is None and "fully_qualified_name" in self.model_fields_set:
            _dict['fullyQualifiedName'] = None

        # set to None if name (nullable) is None
        # and model_fields_set contains the field
        if self.name is None and "name" in self.model_fields_set:
            _dict['name'] = None

        # set to None if scope_name (nullable) is None
        # and model_fields_set contains the field
        if self.scope_name is None and "scope_name" in self.model_fields_set:
            _dict['scopeName'] = None

        # set to None if custom_attributes (nullable) is None
        # and model_fields_set contains the field
        if self.custom_attributes is None and "custom_attributes" in self.model_fields_set:
            _dict['customAttributes'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of Module from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "assembly": Assembly.from_dict(obj["assembly"]) if obj.get("assembly") is not None else None,
            "fullyQualifiedName": obj.get("fullyQualifiedName"),
            "name": obj.get("name"),
            "mdStreamVersion": obj.get("mdStreamVersion"),
            "moduleVersionId": obj.get("moduleVersionId"),
            "scopeName": obj.get("scopeName"),
            "moduleHandle": ModuleHandle.from_dict(obj["moduleHandle"]) if obj.get("moduleHandle") is not None else None,
            "customAttributes": [CustomAttributeData.from_dict(_item) for _item in obj["customAttributes"]] if obj.get("customAttributes") is not None else None,
            "metadataToken": obj.get("metadataToken")
        })
        return _obj

from civitai.models.assembly import Assembly
from civitai.models.custom_attribute_data import CustomAttributeData
# TODO: Rewrite to not use raise_errors
Module.model_rebuild(raise_errors=False)
