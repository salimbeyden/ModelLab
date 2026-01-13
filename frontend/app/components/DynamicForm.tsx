"use client";

import React from 'react';
import Form from '@rjsf/core';
import validator from '@rjsf/validator-ajv8';

interface DynamicFormProps {
    schema: any;
    uiSchema?: any;
    onSubmit: (data: any) => void;
    formData?: any;
}

const DynamicForm: React.FC<DynamicFormProps> = ({ schema, uiSchema, onSubmit, formData }) => {
    return (
        <div className="rjsf-basic">
            <Form
                schema={schema}
                uiSchema={uiSchema}
                validator={validator}
                onSubmit={({ formData }) => onSubmit(formData)}
                formData={formData}
            />
        </div>
    );
};

export default DynamicForm;
