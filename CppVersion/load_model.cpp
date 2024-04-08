//
// Created by hq on 2024/4/3.
//

#include "load_model.h"

#include <google/protobuf/io/coded_stream.h>
#include "saved_model.pb.h" // Generated from your .proto file

using google::protobuf::io::CodedInputStream;
// Assume "input" is a pointer to an array of bytes containing your serialized message

google::protobuf::uint8* buffer = ...;
int size = ...; // Size of the buffer

// Create a CodedInputStream from the buffer
CodedInputStream coded_input(buffer, size);

// Parse the message
MyProtobufMessage message;
if (!message.ParseFromCodedStream(&coded_input)) {
// Handle parsing error
}

Status ReadBinaryProto(Env* env, const string& fname,
                       protobuf::MessageLite* proto) {
    std::unique_ptr<RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
    std::unique_ptr<FileStream> stream(new FileStream(file.get()));
    protobuf::io::CodedInputStream coded_stream(stream.get());

    if (!proto->ParseFromCodedStream(&coded_stream) ||
        !coded_stream.ConsumedEntireMessage()) {
        TF_RETURN_IF_ERROR(stream->status());
        return errors::DataLoss("Can't parse ", fname, " as binary proto");
    }
    return OkStatus();
}
