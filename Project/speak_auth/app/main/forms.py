from wtforms import StringField, SubmitField, ValidationError
from wtforms.validators import DataRequired, Length

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from flask_uploads import AUDIO, IMAGES

from ..models import Person


class UploadAudioFile(FlaskForm):
    audio = FileField(label='音频文件', validators=[DataRequired(), FileAllowed(AUDIO, message='仅支持上传音频')])
    submit = SubmitField(label='提交')


class NewPersonForm(FlaskForm):
    uid = StringField(label='唯一识别吗(UID)', validators=[DataRequired(), Length(1, 64)])
    person_name = StringField(label='姓名', validators=[DataRequired()])
    avatar = FileField(label='头像', validators=[FileAllowed(IMAGES, message='仅支持上传图片')])
    audio = FileField(label='音频文件', validators=[DataRequired(), FileAllowed(AUDIO, message='仅支持上传音频')])
    submit = SubmitField(label='提交')

    def validate_uid(self, field):
        if Person.query.filter_by(uid=field.data).first():
            raise ValidationError('数据库中已有此人信息')


class InputUIDForm(FlaskForm):
    uid = StringField(label='唯一识别码(UID)', validators=[DataRequired(), Length(1, 64)])
    # audio = FileField(label='音频文件', validators=[DataRequired(), FileAllowed(AUDIO, message='仅支持上传音频')])
    submit = SubmitField(label='下一步')

    def validate_uid(self, field):
        if Person.query.filter_by(uid=field.data).first() is None:
            raise ValidationError('数据库中查无此人')


class NewPersonFormV2(FlaskForm):
    uid = StringField(label='唯一识别码(UID)', validators=[DataRequired(), Length(1, 64)])
    person_name = StringField(label='姓名', validators=[DataRequired()])
    avatar = FileField(label='头像', validators=[FileAllowed(IMAGES, message='仅支持上传图片')])
    submit = SubmitField(label='下一步')

    def validate_uid(self, field):
        if Person.query.filter_by(uid=field.data).first():
            raise ValidationError('数据库中已有此人信息')
