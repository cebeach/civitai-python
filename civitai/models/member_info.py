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

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr
from typing import Any, ClassVar, Dict, List, Optional
from civitai.models.member_types import MemberTypes
from typing import Optional, Set
from typing_extensions import Self

class MemberInfo(BaseModel):
    """
    MemberInfo
    """ # noqa: E501
    member_type: Optional[MemberTypes] = Field(default=None, alias="memberType")
    name: Optional[StrictStr] = None
    declaring_type: Optional[Type] = Field(default=None, alias="declaringType")
    reflected_type: Optional[Type] = Field(default=None, alias="reflectedType")
    module: Optional[Module] = None
    custom_attributes: Optional[List[CustomAttributeData]] = Field(default=None, alias="customAttributes")
    is_collectible: Optional[StrictBool] = Field(default=None, alias="isCollectible")
    metadata_token: Optional[StrictInt] = Field(default=None, alias="metadataToken")
    __properties: ClassVar[List[str]] = ["memberType", "name", "declaringType", "reflectedType", "module", "customAttributes", "isCollectible", "metadataToken"]

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
        """Create an instance of MemberInfo from a JSON string"""
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
        """
        excluded_fields: Set[str] = set([
            "name",
            "custom_attributes",
            "is_collectible",
            "metadata_token",
        ])

        _dict = self.model_dump(
            by_alias=True,
            exclude=excluded_fields,
            exclude_none=True,
        )
        # override the default output from pydantic by calling `to_dict()` of declaring_type
        if self.declaring_type:
            _dict['declaringType'] = self.declaring_type.to_dict()
        # override the default output from pydantic by calling `to_dict()` of reflected_type
        if self.reflected_type:
            _dict['reflectedType'] = self.reflected_type.to_dict()
        # override the default output from pydantic by calling `to_dict()` of module
        if self.module:
            _dict['module'] = self.module.to_dict()
        # override the default output from pydantic by calling `to_dict()` of each item in custom_attributes (list)
        _items = []
        if self.custom_attributes:
            for _item in self.custom_attributes:
                if _item:
                    _items.append(_item.to_dict())
            _dict['customAttributes'] = _items
        # set to None if name (nullable) is None
        # and model_fields_set contains the field
        if self.name is None and "name" in self.model_fields_set:
            _dict['name'] = None

        # set to None if custom_attributes (nullable) is None
        # and model_fields_set contains the field
        if self.custom_attributes is None and "custom_attributes" in self.model_fields_set:
            _dict['customAttributes'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of MemberInfo from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate({
            "memberType": obj.get("memberType"),
            "name": obj.get("name"),
            "declaringType": Type.from_dict(obj["declaringType"]) if obj.get("declaringType") is not None else None,
            "reflectedType": Type.from_dict(obj["reflectedType"]) if obj.get("reflectedType") is not None else None,
            "module": Module.from_dict(obj["module"]) if obj.get("module") is not None else None,
            "customAttributes": [CustomAttributeData.from_dict(_item) for _item in obj["customAttributes"]] if obj.get("customAttributes") is not None else None,
            "isCollectible": obj.get("isCollectible"),
            "metadataToken": obj.get("metadataToken")
        })
        return _obj

from civitai.models.custom_attribute_data import CustomAttributeData
from civitai.models.module import Module
from civitai.models.type import Type
# TODO: Rewrite to not use raise_errors
MemberInfo.model_rebuild(raise_errors=False)
