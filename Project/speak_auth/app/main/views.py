import os
import time
from pydub import AudioSegment

from flask import (
        render_template, redirect, url_for,
        flash, request, current_app, session
    )

from . import main
from .forms import UploadAudioFile, NewPersonForm, InputUIDForm, \
    NewPersonFormV2
from .threading_functions import extract_mfcc
from .dtw_auth import random_dtw_number, auth_pipeline
from .. import db, user_img, audio
from ..models import Person, LoginRecord
from ..assist_func import random_string


@main.route('/', methods=['GET', 'POST'])
def index():
    return redirect(url_for('main.new_person'))


"""
新测试
"""
# 新建用户
@main.route('/new_person', methods=['GET', 'POST'])
def new_person():
    form = NewPersonForm()
    if form.validate_on_submit():
        person = Person(uid=form.uid.data,
                        person_name=form.person_name.data,
                        dtw_number=session.get('dtw_number'))
        if form.avatar.data:
            avatar_suffix = os.path.splitext(form.avatar.data.filename)[1]
            avatar_filename = random_string() + avatar_suffix
            user_img.save(form.avatar.data, name=avatar_filename)  # 保存图片
            person.avatar = avatar_filename
        audio_suffix = os.path.splitext(form.audio.data.filename)[1]
        audio_filename = random_string() + audio_suffix
        audio.save(form.audio.data, name=audio_filename)
        extract_mfcc(user_id=person.uid,
                     audio_dir=current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename,
                     save_to=current_app.config['UPLOADED_PERSON_DEST'])
        person.dtw_mfcc = str(person.uid) + '.csv'
        db.session.add(person)
        db.session.commit()
        flash('身份创建成功')
        return redirect(url_for('main.index'))
    session['dtw_number'] = random_dtw_number()
    return render_template("new_person.html", dtw_number=session.get('dtw_number'), form=form)


# 身份验证第一步
@main.route('/authentication/step-1', methods=['GET', 'POST'])
def auth_1st_step():
    form = InputUIDForm()
    if form.validate_on_submit():
        person = Person.query.filter_by(uid=form.uid.data).first_or_404()
        session['dtw_auth_number'] = person.dtw_number
        return redirect(url_for('main.auth_2nd_step', uid=person.uid))
    return render_template("auth_1st_step.html", form=form)


# 身份验证第二步
@main.route('/authentication/step-2/<uid>', methods=['GET', 'POST'])
def auth_2nd_step(uid):
    form = UploadAudioFile()
    person = Person.query.filter_by(uid=uid).first_or_404()
    if form.validate_on_submit():
        audio_suffix = os.path.splitext(form.audio.data.filename)[1]
        audio_filename = random_string() + audio_suffix
        audio.save(form.audio.data, name=audio_filename)
        dtw_result, real_man_result = auth_pipeline(dtw_features_path=
                                                    current_app.config['UPLOADED_PERSON_DEST'] + '/' + person.dtw_mfcc,
                                                    audio_file_path=
                                                    current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename,
                                                    common_data_path=
                                                    current_app.config['GMM_COMMON_DATA'],
                                                    throat_data_path=
                                                    current_app.config['GMM_THROAT_DATA'])
        login_record = LoginRecord(person_id=person.id,
                                   dtw_result=dtw_result,
                                   svm_result=real_man_result)
        db.session.add(login_record)
        db.session.commit()
        if dtw_result and real_man_result:
            flash('恭喜！验证通过')
        else:
            flash('抱歉，验证未通过')
        return redirect(url_for('main.auth_results', uid=person.uid))
    return render_template("auth_2nd_step.html", form=form, dtw_number=person.dtw_number)


# 验证结果页
@main.route('/authentication/results/<uid>', methods=['GET'])
def auth_results(uid):
    person = Person.query.filter_by(uid=uid).first_or_404()
    records = person.login_records.order_by(LoginRecord.timestamp.desc()).all()
    results = [{
        'dtw_result': record.dtw_result,
        'svm_result': record.svm_result,
        'timestamp': record.timestamp,
    }for record in records]
    return render_template("auth_results.html", person=person, results=results, user_img=user_img)


# 身份管理页
@main.route('/authentication/person_admin', methods=['GET', 'POST'])
def person_admin():
    page = request.args.get('page', 1, type=int)
    pagination = Person.query.order_by(Person.id.desc()).paginate(
        page, per_page=current_app.config['SPEAK_AUTH_USER_PER_PAGE'],
        error_out=False
    )
    persons = [{
        'uid': person.id,
        'person_name': person.person_name,
        'dtw_number': person.dtw_number,
        'avatar': person.avatar,
    }for person in pagination.items]
    return render_template("person_admin.html", persons=persons, pagination=pagination,
                           user_img=user_img, endpoint='main.person_admin')


"""
第二版采用浏览器录音功能
"""
@main.route('/v2/new_person/step-1', methods=['GET', 'POST'])
def new_person_v2_1st_step():
    form = NewPersonFormV2()
    if form.validate_on_submit():
        person = Person(uid=form.uid.data,
                        person_name=form.person_name.data,
                        dtw_number=session.get('dtw_number'))
        if form.avatar.data:
            avatar_suffix = os.path.splitext(form.avatar.data.filename)[1]
            avatar_filename = random_string() + avatar_suffix
            user_img.save(form.avatar.data, name=avatar_filename)  # 保存图片
            person.avatar = avatar_filename
        db.session.add(person)
        db.session.commit()
        flash('身份创建成功，现在收集您的语音信息')
        return redirect(url_for('main.new_person_v2_2nd_step', uid=person.uid))
    return render_template("v2/new_person.html", form=form)


@main.route('/v2/new_person/step-2/<uid>', methods=['GET', 'POST'])
def new_person_v2_2nd_step(uid):
    person = Person.query.filter_by(uid=uid).first_or_404()
    return render_template("v2/test_new_person.html", dtw_number=person.dtw_number, uid=person.uid)


@main.route('/v2/new_person/step-3/<uid>', methods=['POST'])
def new_person_v2_3rd_step(uid):
    person = Person.query.filter_by(uid=uid).first_or_404()
    if request.files['file']:
        audio_filename = random_string() + '.mp3'
        audio.save(request.files['file'], name=audio_filename)
        audio_file = AudioSegment.from_mp3(current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename) \
            .set_frame_rate(11025)
        audio_filename_wav = random_string() + '.wav'
        audio_file.export(out_f=current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename_wav, format='wav')
        extract_mfcc(user_id=person.uid,
                     audio_dir=current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename_wav,
                     save_to=current_app.config['UPLOADED_PERSON_DEST'])
        person.dtw_mfcc = str(person.uid) + '.csv'
        db.session.add(person)
        db.session.commit()
        print('MFCCs-OK!')


@main.route('/v2/authentication/step-1', methods=['GET', 'POST'])
def auth_v2_1st_step():
    form = InputUIDForm()
    if form.validate_on_submit():
        person = Person.query.filter_by(uid=form.uid.data).first_or_404()
        return redirect(url_for('main.auth_v2_2nd_step', uid=person.uid))
    return render_template("v2/auth_1st_step.html", form=form)


@main.route('/v2/authentication/step-2/<uid>', methods=['GET', 'POST'])
def auth_v2_2nd_step(uid):
    person = Person.query.filter_by(uid=uid).first_or_404()
    return render_template("v2/test_auth.html", dtw_number=person.dtw_number, uid=person.uid)


@main.route('/v2/authentication/step-3/<uid>', methods=['POST'])
def auth_v2_3rd_step(uid):
    person = Person.query.filter_by(uid=uid).first_or_404()
    if request.files['file']:
        audio_filename = random_string() + '.mp3'
        audio.save(request.files['file'], name=audio_filename)
        audio_file = AudioSegment.from_mp3(current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename)\
            .set_frame_rate(11025)
        audio_filename_wav = random_string() + '.wav'
        audio_file.export(out_f=current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename_wav, format='wav')
        os.remove(current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename)
        dtw_result, real_man_result = auth_pipeline(dtw_features_path=
                                                    current_app.config['UPLOADED_PERSON_DEST'] + '/' + person.dtw_mfcc,
                                                    audio_file_path=
                                                    current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename_wav,
                                                    common_data_path=
                                                    current_app.config['GMM_COMMON_DATA'],
                                                    throat_data_path=
                                                    current_app.config['GMM_THROAT_DATA'])
        login_record = LoginRecord(person_id=person.id,
                                   dtw_result=dtw_result,
                                   svm_result=real_man_result)
        db.session.add(login_record)
        db.session.commit()
        return redirect(url_for('main.jump', uid=person.uid))


@main.route('/jump/<uid>', methods=['GET', 'POST'])
def jump(uid):
    time.sleep(4.0)
    flash('请稍等，正在验证身份')
    return render_template("wait.html", uid=uid)
