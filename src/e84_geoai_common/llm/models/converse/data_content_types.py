import base64
from typing import Any, Literal, Self, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field

from e84_geoai_common.llm.core.llm import (
    Base64ImageContent,
    LLMMediaType,
)

# Converse uses camel case for its variables. Ignore any linting problems with this.
# ruff: noqa: N815


ConverseImageFormat = Literal["jpeg", "png", "gif", "webp"]
ConverseVideoFormat = Literal["mkv", "mov", "mp4", "webm", "flv", "mpeg", "mpg", "wmv", "three_gp"]
# aliasing because Pydantic complains if you define a field like: bytes: bytes = Field(...)
BytesType = bytes


class ConverseTextContent(BaseModel):
    """Converse text context model."""

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str

    def __str__(self) -> str:
        return self.text


# Not defining this as a BaseModel because of name conflict with field "json"
class ConverseJSONContent(TypedDict):
    json: dict[str, Any]


class ConverseDocumentSource(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    bytes: BytesType = Field(description="The raw bytes for the document.")


class ConverseDocumentBlock(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    format: Literal["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]
    name: str
    source: ConverseDocumentSource


class ConverseDocumentContent(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    document: ConverseDocumentBlock


class ConverseImageSource(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    bytes: BytesType = Field(description="The raw bytes for the image.")


class ConverseImageBlock(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    format: ConverseImageFormat
    source: ConverseImageSource


class ConverseImageContent(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    image: ConverseImageBlock

    @classmethod
    def from_b64_image_content(cls, image: Base64ImageContent) -> Self:
        img_format: ConverseImageFormat = cast(ConverseImageFormat, image.media_type.split("/")[-1])
        source = ConverseImageSource(bytes=base64.b64decode(image.data))
        return cls(image=ConverseImageBlock(format=img_format, source=source))

    def to_b64_image_content(self) -> Base64ImageContent:
        media_type: LLMMediaType = cast(
            LLMMediaType,
            f"image/{self.image.format}",
        )
        return Base64ImageContent(
            media_type=media_type,
            data=self.image.source.bytes.decode("utf8"),
        )


class ConverseS3Location(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    uri: str
    bucketOwner: str | None = Field(
        description="If the bucket belongs to another AWS account, specify that account's ID."
    )


class ConverseVideoSource(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    bytes: bytes
    s3Location: ConverseS3Location


class ConverseVideoBlock(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    format: ConverseVideoFormat
    source: ConverseVideoSource


class ConverseVideoContent(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    video: ConverseVideoBlock
